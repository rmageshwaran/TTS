#!/usr/bin/env python3
"""
Test Audio Generation Script for TTS Virtual Microphone

Generates comprehensive test audio files to verify TTS quality and system performance.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import setup_logging, get_logger
from core.tts_engine import TTSEngine
from core.audio_processor import AudioProcessor
from core.text_processor import TextProcessor
from config.config_manager import ConfigManager

# Setup logging
setup_logging(level="INFO", console_output=True)
logger = get_logger(__name__)


class TestAudioGenerator:
    """
    Comprehensive test audio generator for TTS quality verification.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize test audio generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = Path("./test_audio_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.tts_engine = TTSEngine(config)
        self.audio_processor = AudioProcessor(config)
        self.text_processor = TextProcessor(config)
        
        # Test results
        self.test_results = {
            "generated_files": [],
            "total_duration": 0.0,
            "total_size": 0,
            "quality_metrics": {},
            "errors": []
        }
        
        logger.info("TestAudioGenerator initialized")
    
    def generate_comprehensive_tests(self) -> Dict[str, Any]:
        """
        Generate comprehensive test audio files.
        
        Returns:
            Dict with test results
        """
        logger.info("Starting comprehensive test audio generation")
        
        try:
            # Test 1: Basic functionality
            self._test_basic_functionality()
            
            # Test 2: Voice quality tests
            self._test_voice_quality()
            
            # Test 3: Language and accent tests
            self._test_language_variations()
            
            # Test 4: Speed and pacing tests
            self._test_speed_variations()
            
            # Test 5: Emotion and tone tests
            self._test_emotion_variations()
            
            # Test 6: Technical content tests
            self._test_technical_content()
            
            # Test 7: Long-form content tests
            self._test_long_form_content()
            
            # Test 8: Voice cloning tests (if available)
            self._test_voice_cloning()
            
            # Generate summary report
            self._generate_summary_report()
            
            logger.info("Comprehensive test audio generation completed")
            return self.test_results
            
        except Exception as e:
            logger.error(f"Test generation failed: {e}")
            self.test_results["errors"].append(str(e))
            return self.test_results
    
    def _test_basic_functionality(self):
        """Test basic TTS functionality."""
        logger.info("Testing basic functionality...")
        
        basic_tests = [
            {
                "name": "simple_greeting",
                "text": "Hello, this is a test of the text-to-speech system.",
                "description": "Simple greeting to test basic functionality"
            },
            {
                "name": "numbers_test",
                "text": "The numbers are: one, two, three, four, five, six, seven, eight, nine, ten.",
                "description": "Number pronunciation test"
            },
            {
                "name": "alphabet_test",
                "text": "The alphabet: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z.",
                "description": "Alphabet pronunciation test"
            },
            {
                "name": "punctuation_test",
                "text": "This sentence has a comma, period. Question mark? Exclamation point! Quotation marks: \"Hello world\".",
                "description": "Punctuation handling test"
            }
        ]
        
        for test in basic_tests:
            self._generate_test_audio(test)
    
    def _test_voice_quality(self):
        """Test voice quality characteristics."""
        logger.info("Testing voice quality...")
        
        quality_tests = [
            {
                "name": "clarity_test",
                "text": "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet at least once.",
                "description": "Voice clarity and articulation test"
            },
            {
                "name": "naturalness_test",
                "text": "I'm speaking in a natural, conversational tone. This should sound like a real person talking, not a robot.",
                "description": "Natural speech patterns test"
            },
            {
                "name": "prosody_test",
                "text": "The weather today is sunny and warm. I'm feeling quite happy about it!",
                "description": "Prosody and intonation test"
            },
            {
                "name": "breathing_test",
                "text": "This is a longer sentence that tests natural breathing patterns and pauses in speech.",
                "description": "Natural breathing and pauses test"
            }
        ]
        
        for test in quality_tests:
            self._generate_test_audio(test)
    
    def _test_language_variations(self):
        """Test different languages and accents."""
        logger.info("Testing language variations...")
        
        language_tests = [
            {
                "name": "english_standard",
                "text": "This is standard English pronunciation with clear enunciation.",
                "description": "Standard English pronunciation"
            },
            {
                "name": "english_formal",
                "text": "Ladies and gentlemen, it is with great pleasure that I address you today.",
                "description": "Formal English speech"
            },
            {
                "name": "english_casual",
                "text": "Hey there! How's it going? This is just a casual conversation.",
                "description": "Casual English speech"
            },
            {
                "name": "technical_terms",
                "text": "The algorithm processes data through multiple neural network layers.",
                "description": "Technical terminology pronunciation"
            }
        ]
        
        for test in language_tests:
            self._generate_test_audio(test)
    
    def _test_speed_variations(self):
        """Test different speaking speeds."""
        logger.info("Testing speed variations...")
        
        speed_tests = [
            {
                "name": "slow_speech",
                "text": "This is spoken very slowly and deliberately for clarity.",
                "description": "Slow, deliberate speech",
                "speed": 0.8
            },
            {
                "name": "normal_speech",
                "text": "This is normal conversational speed, neither too fast nor too slow.",
                "description": "Normal conversational speed",
                "speed": 1.0
            },
            {
                "name": "fast_speech",
                "text": "This is spoken quickly but still clearly understandable.",
                "description": "Fast but clear speech",
                "speed": 1.2
            }
        ]
        
        for test in speed_tests:
            self._generate_test_audio(test)
    
    def _test_emotion_variations(self):
        """Test different emotional tones."""
        logger.info("Testing emotion variations...")
        
        emotion_tests = [
            {
                "name": "happy_emotion",
                "text": "I'm so excited and happy about this amazing news!",
                "description": "Happy and excited tone"
            },
            {
                "name": "calm_emotion",
                "text": "Everything is peaceful and calm. Take a deep breath and relax.",
                "description": "Calm and peaceful tone"
            },
            {
                "name": "serious_emotion",
                "text": "This is a serious matter that requires our immediate attention.",
                "description": "Serious and concerned tone"
            },
            {
                "name": "enthusiastic_emotion",
                "text": "This is absolutely fantastic! I can't believe how wonderful this is!",
                "description": "Enthusiastic and energetic tone"
            }
        ]
        
        for test in emotion_tests:
            self._generate_test_audio(test)
    
    def _test_technical_content(self):
        """Test technical and specialized content."""
        logger.info("Testing technical content...")
        
        technical_tests = [
            {
                "name": "scientific_terms",
                "text": "The deoxyribonucleic acid molecule contains genetic information.",
                "description": "Scientific terminology pronunciation"
            },
            {
                "name": "mathematical_terms",
                "text": "The quadratic equation x squared plus bx plus c equals zero.",
                "description": "Mathematical expression pronunciation"
            },
            {
                "name": "programming_terms",
                "text": "The function returns a boolean value indicating success or failure.",
                "description": "Programming terminology pronunciation"
            },
            {
                "name": "medical_terms",
                "text": "The patient exhibits symptoms of hypertension and tachycardia.",
                "description": "Medical terminology pronunciation"
            }
        ]
        
        for test in technical_tests:
            self._generate_test_audio(test)
    
    def _test_long_form_content(self):
        """Test longer content for consistency."""
        logger.info("Testing long-form content...")
        
        long_form_tests = [
            {
                "name": "story_paragraph",
                "text": "Once upon a time, in a distant land, there lived a wise old wizard who possessed the power to transform ordinary objects into magical ones. He spent his days in a tall tower, surrounded by ancient books and mysterious artifacts. People from far and wide would come to seek his wisdom and magical assistance.",
                "description": "Story paragraph for consistency testing"
            },
            {
                "name": "news_article",
                "text": "Breaking news: Scientists have made a groundbreaking discovery in renewable energy technology. The new solar panel design achieves unprecedented efficiency levels while reducing manufacturing costs by forty percent. This development could revolutionize the clean energy industry and accelerate the transition to sustainable power sources worldwide.",
                "description": "News article style content"
            },
            {
                "name": "educational_content",
                "text": "Today we'll learn about the water cycle, a fundamental process that sustains life on Earth. Water evaporates from oceans, lakes, and rivers, forming clouds in the atmosphere. When conditions are right, the water condenses and falls back to Earth as precipitation, completing the cycle.",
                "description": "Educational content style"
            }
        ]
        
        for test in long_form_tests:
            self._generate_test_audio(test)
    
    def _test_voice_cloning(self):
        """Test voice cloning capabilities if available."""
        logger.info("Testing voice cloning capabilities...")
        
        # Check if voice cloning is available
        if hasattr(self.tts_engine, 'get_available_voices'):
            voices = self.tts_engine.get_available_voices()
            if voices and len(voices) > 1:
                for i, voice in enumerate(voices[:3]):  # Test first 3 voices
                    test = {
                        "name": f"voice_clone_{voice.get('name', f'voice_{i}')}",
                        "text": "This is a test of voice cloning capabilities with different voice characteristics.",
                        "description": f"Voice cloning test with {voice.get('name', f'voice_{i}')}",
                        "voice_id": voice.get('id')
                    }
                    self._generate_test_audio(test)
            else:
                logger.info("Voice cloning not available or only one voice found")
        else:
            logger.info("Voice cloning interface not available")
    
    def _generate_test_audio(self, test_config: Dict[str, Any]) -> bool:
        """
        Generate a single test audio file.
        
        Args:
            test_config: Test configuration dictionary
            
        Returns:
            bool: True if generation successful
        """
        try:
            test_name = test_config["name"]
            text = test_config["text"]
            description = test_config["description"]
            
            logger.info(f"Generating test: {test_name}")
            
            # Process text
            text_result = self.text_processor.process(text)
            processed_text = text_result.processed_text
            
            # Generate speech
            start_time = time.time()
            tts_result = self.tts_engine.generate_speech(processed_text)
            generation_time = time.time() - start_time
            
            # Extract audio data from TTS result
            speech_data = tts_result.audio_data
            
            # Process audio
            audio_result = self.audio_processor.process(speech_data)
            processed_audio = audio_result.audio_data
            
            # Save to file
            output_file = self.output_dir / f"{test_name}.wav"
            with open(output_file, 'wb') as f:
                f.write(processed_audio)
            
            # Calculate metrics
            file_size = len(processed_audio)
            duration = file_size / (self.config.get('audio', {}).get('sample_rate', 24000) * 2)  # Approximate duration
            
            # Record test result
            test_result = {
                "name": test_name,
                "description": description,
                "text": text,
                "file_path": str(output_file),
                "file_size": file_size,
                "duration": duration,
                "generation_time": generation_time,
                "characters": len(text),
                "characters_per_second": len(text) / generation_time if generation_time > 0 else 0
            }
            
            self.test_results["generated_files"].append(test_result)
            self.test_results["total_duration"] += duration
            self.test_results["total_size"] += file_size
            
            logger.info(f"Generated {test_name}: {duration:.2f}s, {file_size} bytes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate test {test_config.get('name', 'unknown')}: {e}")
            self.test_results["errors"].append(f"{test_config.get('name', 'unknown')}: {e}")
            return False
    
    def _generate_summary_report(self):
        """Generate a comprehensive summary report."""
        logger.info("Generating summary report...")
        
        # Calculate quality metrics
        if self.test_results["generated_files"]:
            total_files = len(self.test_results["generated_files"])
            avg_generation_time = sum(f["generation_time"] for f in self.test_results["generated_files"]) / total_files
            avg_chars_per_second = sum(f["characters_per_second"] for f in self.test_results["generated_files"]) / total_files
            
            self.test_results["quality_metrics"] = {
                "total_files_generated": total_files,
                "total_duration_minutes": self.test_results["total_duration"] / 60,
                "total_size_mb": self.test_results["total_size"] / (1024 * 1024),
                "average_generation_time": avg_generation_time,
                "average_characters_per_second": avg_chars_per_second,
                "success_rate": (total_files - len(self.test_results["errors"])) / total_files * 100
            }
        
        # Save detailed report
        report_file = self.output_dir / "test_report.json"
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        # Save summary text report
        summary_file = self.output_dir / "test_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("TTS Test Audio Generation Summary\n")
            f.write("=" * 50 + "\n\n")
            
            if self.test_results["quality_metrics"]:
                metrics = self.test_results["quality_metrics"]
                f.write(f"Files Generated: {metrics['total_files_generated']}\n")
                f.write(f"Total Duration: {metrics['total_duration_minutes']:.2f} minutes\n")
                f.write(f"Total Size: {metrics['total_size_mb']:.2f} MB\n")
                f.write(f"Average Generation Time: {metrics['average_generation_time']:.2f} seconds\n")
                f.write(f"Average Speed: {metrics['average_characters_per_second']:.1f} chars/sec\n")
                f.write(f"Success Rate: {metrics['success_rate']:.1f}%\n\n")
            
            f.write("Generated Files:\n")
            f.write("-" * 30 + "\n")
            for file_info in self.test_results["generated_files"]:
                f.write(f"{file_info['name']}: {file_info['duration']:.2f}s, {file_info['file_size']} bytes\n")
            
            if self.test_results["errors"]:
                f.write("\nErrors:\n")
                f.write("-" * 30 + "\n")
                for error in self.test_results["errors"]:
                    f.write(f"- {error}\n")
        
        logger.info(f"Summary report saved to {summary_file}")
        logger.info(f"Detailed report saved to {report_file}")
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self.tts_engine, 'cleanup'):
                self.tts_engine.cleanup()
            logger.info("TestAudioGenerator cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


def main():
    """Main function for test audio generation."""
    parser = argparse.ArgumentParser(description="Generate Test Audio Files for TTS Quality Verification")
    parser.add_argument("--config", default="./config/config.yaml", 
                       help="Configuration file path")
    parser.add_argument("--output-dir", default="./test_audio_output",
                       help="Output directory for test files")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick test with fewer samples")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config_manager = ConfigManager(args.config)
        config_obj = config_manager.load()
        config = config_manager._config_to_dict(config_obj)
        
        # Override output directory if specified
        if args.output_dir:
            config["test_audio"] = config.get("test_audio", {})
            config["test_audio"]["output_dir"] = args.output_dir
        
        # Create generator
        generator = TestAudioGenerator(config)
        
        # Generate tests
        if args.quick:
            logger.info("Running quick test...")
            # Run only basic tests for quick verification
            generator._test_basic_functionality()
            generator._test_voice_quality()
            generator._generate_summary_report()
        else:
            logger.info("Running comprehensive test...")
            results = generator.generate_comprehensive_tests()
        
        # Print summary
        if generator.test_results["quality_metrics"]:
            metrics = generator.test_results["quality_metrics"]
            print(f"\n🎉 Test Generation Complete!")
            print(f"📁 Files Generated: {metrics['total_files_generated']}")
            print(f"⏱️  Total Duration: {metrics['total_duration_minutes']:.2f} minutes")
            print(f"💾 Total Size: {metrics['total_size_mb']:.2f} MB")
            print(f"⚡ Success Rate: {metrics['success_rate']:.1f}%")
            print(f"📊 Average Speed: {metrics['average_characters_per_second']:.1f} chars/sec")
            print(f"📂 Output Directory: {args.output_dir}")
        
        # Cleanup
        generator.cleanup()
        
    except Exception as e:
        logger.error(f"Test generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
