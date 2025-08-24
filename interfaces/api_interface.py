#!/usr/bin/env python3
"""
REST API Interface for TTS Virtual Microphone

Provides RESTful API endpoints for text-to-speech processing
and virtual microphone routing functionality.
"""

import asyncio
import logging
import uvicorn
from typing import Optional, Dict, Any, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import core components
from core.text_processor import TextProcessor
from core.tts_engine import TTSEngine
from core.audio_processor import AudioProcessor
from core.virtual_audio import VirtualAudioInterface

logger = logging.getLogger(__name__)


# Request/Response Models
class TTSRequest(BaseModel):
    """Request model for text-to-speech conversion."""
    text: str = Field(..., description="Text to convert to speech", min_length=1, max_length=10000)
    speaker_id: int = Field(0, description="Speaker ID for voice selection", ge=0)
    temperature: float = Field(1.0, description="Voice temperature setting", ge=0.1, le=2.0)
    output_format: str = Field("wav", description="Audio output format")


class TTSResponse(BaseModel):
    """Response model for text-to-speech conversion."""
    success: bool
    message: str
    audio_length_ms: Optional[int] = None
    audio_size_bytes: Optional[int] = None
    processing_time_ms: Optional[float] = None
    device_used: Optional[str] = None
    error_details: Optional[str] = None


class SystemStatus(BaseModel):
    """System status response model."""
    status: str
    tts_engine: Dict[str, Any]
    audio_processor: Dict[str, Any]
    virtual_audio: Dict[str, Any]
    available_devices: List[Dict[str, Any]]


class HealthCheck(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    version: str
    components: Dict[str, str]


class APIInterface:
    """
    REST API Interface for TTS Virtual Microphone.
    
    Provides RESTful endpoints for text-to-speech processing,
    system status monitoring, and configuration management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize API interface.
        
        Args:
            config: Application configuration dictionary
        """
        self.config = config
        self.app = FastAPI(
            title="TTS Virtual Microphone API",
            description="Convert text to speech and route to virtual microphone",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize core components
        try:
            # Convert Config object to dictionary for component initialization
            if hasattr(config, '__dict__'):
                config_dict = self._config_to_dict(config)
            else:
                config_dict = config
            
            self.text_processor = TextProcessor(config_dict)
            self.tts_engine = TTSEngine(config_dict)
            self.audio_processor = AudioProcessor(config_dict)
            self.virtual_audio = VirtualAudioInterface(config_dict)
            
            logger.info("API interface components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize API components: {e}")
            raise
        
        # Setup routes
        self._setup_routes()
    
    def _config_to_dict(self, config) -> Dict[str, Any]:
        """
        Convert Config object to dictionary format expected by components.
        
        Args:
            config: Config dataclass object
            
        Returns:
            Dictionary representation of config
        """
        from dataclasses import asdict
        
        try:
            if hasattr(config, '__dataclass_fields__'):
                return asdict(config)
            else:
                # Already a dictionary or dictionary-like
                return config
        except Exception as e:
            logger.warning(f"Failed to convert config to dict: {e}")
            # Return basic config structure
            return {
                'application': {'log_level': 'INFO'},
                'tts': {'default_engine': 'demo_tts', 'context_length': 4},
                'audio': {'sample_rate': 24000, 'channels': 1, 'format': 'wav'},
                'virtual_audio': {'device_name': 'CABLE Input', 'auto_detect': True}
            }
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint with API information."""
            return {
                "name": "TTS Virtual Microphone API",
                "version": "1.0.0",
                "description": "Convert text to speech and route to virtual microphone",
                "docs": "/docs",
                "health": "/health"
            }
        
        @self.app.get("/health", response_model=HealthCheck)
        async def health_check():
            """Health check endpoint."""
            try:
                # Test component availability
                components = {
                    "text_processor": "healthy",
                    "tts_engine": "healthy", 
                    "audio_processor": "healthy",
                    "virtual_audio": "healthy"
                }
                
                # Quick component checks
                if hasattr(self.tts_engine, 'get_status'):
                    tts_status = self.tts_engine.get_status()
                    if not tts_status.get('available', True):
                        components["tts_engine"] = "unavailable"
                
                if hasattr(self.virtual_audio, 'get_status'):
                    audio_status = self.virtual_audio.get_status()
                    if not audio_status.get('available', True):
                        components["virtual_audio"] = "degraded"
                
                overall_status = "healthy"
                if "unavailable" in components.values():
                    overall_status = "unhealthy"
                elif "degraded" in components.values():
                    overall_status = "degraded"
                
                return HealthCheck(
                    status=overall_status,
                    timestamp=str(asyncio.get_event_loop().time()),
                    version="1.0.0",
                    components=components
                )
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return HealthCheck(
                    status="unhealthy",
                    timestamp=str(asyncio.get_event_loop().time()),
                    version="1.0.0",
                    components={"error": str(e)}
                )
        
        @self.app.post("/tts", response_model=TTSResponse)
        async def convert_text_to_speech(request: TTSRequest):
            """
            Convert text to speech and route to virtual microphone.
            
            Args:
                request: TTS request parameters
                
            Returns:
                TTSResponse: Processing result and metadata
            """
            start_time = asyncio.get_event_loop().time()
            
            try:
                logger.info(f"Processing TTS request: {len(request.text)} characters")
                
                # Step 1: Text Processing
                processed_result = self.text_processor.process(request.text)
                
                if hasattr(processed_result, 'processed_text'):
                    processed_text = processed_result.processed_text
                else:
                    processed_text = str(processed_result)
                
                # Step 2: TTS Generation
                tts_result = self.tts_engine.generate_speech(
                    processed_text,
                    speaker_id=request.speaker_id
                )
                
                if not tts_result.success:
                    raise HTTPException(
                        status_code=500,
                        detail=f"TTS generation failed: {tts_result.error_message}"
                    )
                
                # Step 3: Audio Processing
                audio_result = self.audio_processor.process(tts_result.audio_data)
                
                if not audio_result.success:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Audio processing failed: {audio_result.error_message}"
                    )
                
                # Step 4: Virtual Audio Routing
                device_used = "unknown"
                
                # Try VB-Cable first if available
                if self.virtual_audio.is_vb_cable_available():
                    routing_result = self.virtual_audio.route_via_vb_cable(audio_result.audio_data)
                    if routing_result.success:
                        device_used = getattr(routing_result, 'device_used', 'VB-Cable')
                    else:
                        # Fall back to standard routing
                        routing_result = self.virtual_audio.route_audio(audio_result.audio_data)
                        device_used = getattr(routing_result, 'device_used', 'default')
                else:
                    routing_result = self.virtual_audio.route_audio(audio_result.audio_data)
                    device_used = getattr(routing_result, 'device_used', 'default')
                
                if not routing_result.success:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Audio routing failed: {getattr(routing_result, 'error_message', 'Unknown error')}"
                    )
                
                # Calculate metrics
                processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
                audio_size = len(audio_result.audio_data)
                
                # Estimate audio length (rough calculation)
                if hasattr(self.config, 'audio'):
                    sample_rate = getattr(self.config.audio, 'sample_rate', 24000)
                    channels = getattr(self.config.audio, 'channels', 1)
                else:
                    sample_rate = config_dict.get('audio', {}).get('sample_rate', 24000)
                    channels = config_dict.get('audio', {}).get('channels', 1)
                bytes_per_sample = 2  # 16-bit audio
                audio_length_ms = int((audio_size / (sample_rate * channels * bytes_per_sample)) * 1000)
                
                return TTSResponse(
                    success=True,
                    message="Text successfully converted to speech and routed to virtual microphone",
                    audio_length_ms=audio_length_ms,
                    audio_size_bytes=audio_size,
                    processing_time_ms=processing_time,
                    device_used=device_used
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"TTS API error: {e}")
                processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
                
                return TTSResponse(
                    success=False,
                    message="Internal server error during TTS processing",
                    processing_time_ms=processing_time,
                    error_details=str(e)
                )
        
        @self.app.post("/tts/file", response_model=TTSResponse)
        async def convert_file_to_speech(file: UploadFile = File(...), speaker_id: int = 0, temperature: float = 1.0):
            """
            Convert text file to speech and route to virtual microphone.
            
            Args:
                file: Text file to process
                speaker_id: Speaker ID for voice selection
                temperature: Voice temperature setting
                
            Returns:
                TTSResponse: Processing result and metadata
            """
            try:
                # Read file content
                content = await file.read()
                
                # Try to decode as UTF-8, fall back to latin1
                try:
                    text = content.decode('utf-8').strip()
                except UnicodeDecodeError:
                    text = content.decode('latin1').strip()
                
                if not text:
                    raise HTTPException(status_code=400, detail="File is empty or contains no readable text")
                
                # Process as regular TTS request
                request = TTSRequest(
                    text=text,
                    speaker_id=speaker_id,
                    temperature=temperature  # Note: temperature not used in current TTS implementation
                )
                
                return await convert_text_to_speech(request)
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"File TTS API error: {e}")
                return TTSResponse(
                    success=False,
                    message="Failed to process uploaded file",
                    error_details=str(e)
                )
        
        @self.app.get("/status", response_model=SystemStatus)
        async def get_system_status():
            """
            Get comprehensive system status.
            
            Returns:
                SystemStatus: Current system status and component information
            """
            try:
                # Get TTS engine status
                tts_status = {"status": "available", "loaded_plugins": []}
                if hasattr(self.tts_engine, 'get_status'):
                    tts_status = self.tts_engine.get_status()
                
                # Get audio processor status  
                audio_proc_status = {"status": "available"}
                if hasattr(self.audio_processor, 'get_status'):
                    audio_proc_status = self.audio_processor.get_status()
                
                # Get virtual audio status
                virtual_audio_status = {"status": "available", "vb_cable_available": False}
                if hasattr(self.virtual_audio, 'get_status'):
                    virtual_audio_status = self.virtual_audio.get_status()
                    virtual_audio_status["vb_cable_available"] = self.virtual_audio.is_vb_cable_available()
                
                # Get available devices
                available_devices = []
                if hasattr(self.virtual_audio, 'get_available_devices'):
                    available_devices = self.virtual_audio.get_available_devices()
                
                # Determine overall status
                overall_status = "healthy"
                if (tts_status.get("status") != "available" or 
                    audio_proc_status.get("status") != "available" or
                    virtual_audio_status.get("status") != "available"):
                    overall_status = "degraded"
                
                return SystemStatus(
                    status=overall_status,
                    tts_engine=tts_status,
                    audio_processor=audio_proc_status,
                    virtual_audio=virtual_audio_status,
                    available_devices=available_devices
                )
                
            except Exception as e:
                logger.error(f"Status API error: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get system status: {e}")
        
        @self.app.get("/devices")
        async def list_audio_devices():
            """
            List available audio devices.
            
            Returns:
                List of available audio devices
            """
            try:
                if hasattr(self.virtual_audio, 'get_available_devices'):
                    devices = self.virtual_audio.get_available_devices()
                    return {"devices": devices}
                else:
                    return {"devices": [], "message": "Device listing not available"}
                    
            except Exception as e:
                logger.error(f"Device listing API error: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to list devices: {e}")
    
    def start(self, port: int = 8080, host: str = "0.0.0.0"):
        """
        Start the API server.
        
        Args:
            port: Port to run the server on
            host: Host to bind to
        """
        try:
            logger.info(f"Starting API server on {host}:{port}")
            
            # Configure uvicorn
            config = uvicorn.Config(
                app=self.app,
                host=host,
                port=port,
                log_level="info",
                reload=False
            )
            
            server = uvicorn.Server(config)
            
            # Run the server
            server.run()
            
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            raise
    
    async def start_async(self, port: int = 8080, host: str = "0.0.0.0"):
        """
        Start the API server asynchronously.
        
        Args:
            port: Port to run the server on
            host: Host to bind to
        """
        try:
            logger.info(f"Starting async API server on {host}:{port}")
            
            config = uvicorn.Config(
                app=self.app,
                host=host,
                port=port,
                log_level="info",
                reload=False
            )
            
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            logger.error(f"Failed to start async API server: {e}")
            raise
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'virtual_audio'):
                self.virtual_audio.cleanup()
            if hasattr(self, 'tts_engine'):
                self.tts_engine.cleanup()
            logger.info("API interface cleaned up")
        except Exception as e:
            logger.error(f"API cleanup error: {e}")
