#!/usr/bin/env python3
"""
Background Model Download Manager

Runs model downloads in the background with status monitoring and resume capability.
"""

import os
import sys
import time
import json
import threading
import signal
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import setup_logging, get_logger

# Setup logging
setup_logging(level="INFO", console_output=True)
logger = get_logger(__name__)


class BackgroundDownloadManager:
    """
    Manages background downloads with status tracking and resume capability.
    """
    
    def __init__(self, status_file: str = "./models/download_status.json"):
        self.status_file = Path(status_file)
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self.download_process = None
        self.is_running = False
        
        # Load existing status
        self.status = self._load_status()
    
    def _load_status(self) -> Dict[str, Any]:
        """Load download status from file."""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load status: {e}")
        
        return {
            "downloads": {},
            "active_download": None,
            "last_update": None
        }
    
    def _save_status(self):
        """Save download status to file."""
        try:
            self.status["last_update"] = time.time()
            with open(self.status_file, 'w') as f:
                json.dump(self.status, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save status: {e}")
    
    def start_download(self, model_name: str = "sesame_csm") -> bool:
        """
        Start background download of model.
        
        Args:
            model_name: Name of model to download
            
        Returns:
            bool: True if download started successfully
        """
        try:
            if self.is_running:
                logger.warning("Download already in progress")
                return False
            
            logger.info(f"Starting background download of {model_name}")
            
            # Update status
            self.status["active_download"] = {
                "model": model_name,
                "start_time": time.time(),
                "status": "starting"
            }
            self.status["downloads"][model_name] = {
                "status": "downloading",
                "start_time": time.time(),
                "progress": 0,
                "last_update": time.time()
            }
            self._save_status()
            
            # Start download process
            download_script = project_root / "scripts" / "download_models.py"
            cmd = [sys.executable, str(download_script), "--model", model_name]
            
            self.download_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.is_running = True
            
            # Start monitoring thread
            monitor_thread = threading.Thread(
                target=self._monitor_download,
                args=(model_name,),
                daemon=True
            )
            monitor_thread.start()
            
            logger.info(f"Background download started for {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start download: {e}")
            return False
    
    def _monitor_download(self, model_name: str):
        """Monitor download progress."""
        try:
            while self.is_running and self.download_process:
                # Check if process is still running
                if self.download_process.poll() is not None:
                    # Process finished
                    stdout, stderr = self.download_process.communicate()
                    
                    if self.download_process.returncode == 0:
                        logger.info(f"Download completed successfully: {model_name}")
                        self.status["downloads"][model_name]["status"] = "completed"
                        self.status["downloads"][model_name]["progress"] = 100
                    else:
                        logger.error(f"Download failed: {model_name}")
                        logger.error(f"Error: {stderr}")
                        self.status["downloads"][model_name]["status"] = "failed"
                    
                    self.status["active_download"] = None
                    self.is_running = False
                    self._save_status()
                    break
                
                # Update status
                self.status["downloads"][model_name]["last_update"] = time.time()
                self._save_status()
                
                time.sleep(5)  # Check every 5 seconds
                
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            self.status["downloads"][model_name]["status"] = "error"
            self.is_running = False
            self._save_status()
    
    def stop_download(self) -> bool:
        """
        Stop current download.
        
        Returns:
            bool: True if stopped successfully
        """
        try:
            if not self.is_running or not self.download_process:
                logger.warning("No download in progress")
                return False
            
            logger.info("Stopping download...")
            
            # Terminate process
            self.download_process.terminate()
            
            # Wait for termination
            try:
                self.download_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Force killing download process")
                self.download_process.kill()
            
            self.is_running = False
            
            # Update status
            if self.status["active_download"]:
                model_name = self.status["active_download"]["model"]
                self.status["downloads"][model_name]["status"] = "stopped"
                self.status["active_download"] = None
                self._save_status()
            
            logger.info("Download stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop download: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current download status.
        
        Returns:
            Dict with download status information
        """
        try:
            # Reload status from file
            self.status = self._load_status()
            
            # Add runtime information
            status = self.status.copy()
            status["is_running"] = self.is_running
            
            if self.download_process:
                status["process_id"] = self.download_process.pid
                status["process_returncode"] = self.download_process.poll()
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return {"error": str(e)}
    
    def resume_download(self, model_name: str = "sesame_csm") -> bool:
        """
        Resume interrupted download.
        
        Args:
            model_name: Name of model to resume
            
        Returns:
            bool: True if resumed successfully
        """
        try:
            if self.is_running:
                logger.warning("Download already in progress")
                return False
            
            # Check if model was previously downloading
            if model_name in self.status["downloads"]:
                model_status = self.status["downloads"][model_name]["status"]
                if model_status in ["stopped", "failed", "error"]:
                    logger.info(f"Resuming download of {model_name}")
                    return self.start_download(model_name)
                elif model_status == "completed":
                    logger.info(f"Model {model_name} already downloaded")
                    return True
                else:
                    logger.warning(f"Model {model_name} status: {model_status}")
                    return False
            else:
                logger.info(f"Starting new download of {model_name}")
                return self.start_download(model_name)
                
        except Exception as e:
            logger.error(f"Failed to resume download: {e}")
            return False


def main():
    """Main function for background download manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Background Model Download Manager")
    parser.add_argument("action", choices=["start", "stop", "status", "resume"],
                       help="Action to perform")
    parser.add_argument("--model", default="sesame_csm",
                       help="Model name (default: sesame_csm)")
    parser.add_argument("--status-file", default="./models/download_status.json",
                       help="Status file path")
    
    args = parser.parse_args()
    
    # Create manager
    manager = BackgroundDownloadManager(args.status_file)
    
    try:
        if args.action == "start":
            success = manager.start_download(args.model)
            if success:
                print(f"✅ Started background download of {args.model}")
                sys.exit(0)
            else:
                print(f"❌ Failed to start download of {args.model}")
                sys.exit(1)
        
        elif args.action == "stop":
            success = manager.stop_download()
            if success:
                print("✅ Download stopped")
                sys.exit(0)
            else:
                print("❌ Failed to stop download")
                sys.exit(1)
        
        elif args.action == "status":
            status = manager.get_status()
            print(json.dumps(status, indent=2))
            sys.exit(0)
        
        elif args.action == "resume":
            success = manager.resume_download(args.model)
            if success:
                print(f"✅ Resumed download of {args.model}")
                sys.exit(0)
            else:
                print(f"❌ Failed to resume download of {args.model}")
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        manager.stop_download()
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
