"""
Unit tests for TextProcessor module.

Tests text validation, normalization, SSML processing, and language detection.
"""

import pytest
import numpy as np
from unittest.mock import patch, Mock

from core.text_processor import TextProcessor, TextProcessingResult


class TestTextProcessor:
    """Test cases for TextProcessor class."""
    
    def test_initialization(self, test_config):
        """Test TextProcessor initialization."""
        config_dict = {"text": test_config.text.__dict__}
        processor = TextProcessor(config_dict)
        
        assert processor.max_length == test_config.text.max_length
        assert processor.min_length == test_config.text.min_length
        assert processor.enable_ssml == test_config.text.enable_ssml
    
    def test_validate_valid_text(self, text_processor):
        """Test validation with valid text."""
        assert text_processor.validate("Hello world") is True
        assert text_processor.validate("This is a valid text.") is True
    
    def test_validate_invalid_text(self, text_processor):
        """Test validation with invalid text."""
        # Empty text
        assert text_processor.validate("") is False
        assert text_processor.validate(None) is False
        
        # Whitespace only
        assert text_processor.validate("   ") is False
        assert text_processor.validate("\n\t") is False
        
        # Too long text
        long_text = "x" * (text_processor.max_length + 1)
        assert text_processor.validate(long_text) is False
    
    def test_normalize_basic(self, text_processor):
        """Test basic text normalization."""
        # Multiple spaces
        result = text_processor.normalize("Hello    world")
        assert result == "Hello world"
        
        # Excessive punctuation
        result = text_processor.normalize("Hello!!! World...")
        assert result == "Hello... World..."
        
        # Leading/trailing whitespace
        result = text_processor.normalize("  Hello world  ")
        assert result == "Hello world"
    
    def test_normalize_numbers(self, text_processor):
        """Test number conversion to words."""
        # Basic numbers
        result = text_processor.normalize("I have 5 apples")
        assert "five" in result
        
        result = text_processor.normalize("The year is 2023")
        assert "2023" in result  # Large numbers remain as-is in basic implementation
    
    def test_normalize_emails_and_urls(self, text_processor):
        """Test email and URL normalization."""
        # Email normalization
        result = text_processor.normalize("Contact me at test@example.com")
        assert "at" in result and "dot" in result
        
        # URL normalization
        result = text_processor.normalize("Visit https://example.com")
        assert "URL" in result
    
    def test_process_ssml_enabled(self, text_processor):
        """Test SSML processing when enabled."""
        ssml_text = '<speak>Hello <break time="1s"/> world</speak>'
        result = text_processor.process_ssml(ssml_text)
        
        # Should keep SSML tags when enabled
        assert "<speak>" in result
        assert "<break" in result
    
    def test_process_ssml_disabled(self, test_config):
        """Test SSML processing when disabled."""
        config_dict = {"text": {**test_config.text.__dict__, "enable_ssml": False}}
        processor = TextProcessor(config_dict)
        
        ssml_text = '<speak>Hello <break time="1s"/> world</speak>'
        result = processor.process_ssml(ssml_text)
        
        # Should remove SSML tags when disabled
        assert "<speak>" not in result
        assert "<break" not in result
        assert "Hello world" in result
    
    def test_detect_language(self, text_processor):
        """Test language detection."""
        # English
        assert text_processor.detect_language("Hello world") == "en"
        
        # French (basic heuristic)
        assert text_processor.detect_language("Bonjour le monde") == "fr"
        
        # Spanish (basic heuristic)
        assert text_processor.detect_language("Hola el mundo") == "es"
        
        # German (basic heuristic)
        assert text_processor.detect_language("Hallo die Welt") == "de"
    
    def test_chunk_text_short(self, text_processor):
        """Test text chunking with short text."""
        short_text = "Hello world"
        chunks = text_processor.chunk_text(short_text, max_chunk_size=100)
        
        assert len(chunks) == 1
        assert chunks[0] == short_text
    
    def test_chunk_text_long(self, text_processor):
        """Test text chunking with long text."""
        # Create long text with sentences
        sentences = ["This is sentence one.", "This is sentence two.", "This is sentence three."]
        long_text = " ".join(sentences)
        
        chunks = text_processor.chunk_text(long_text, max_chunk_size=30)
        
        assert len(chunks) > 1
        # Verify no chunk is too long
        for chunk in chunks:
            assert len(chunk) <= 30
    
    def test_chunk_text_very_long_sentence(self, text_processor):
        """Test chunking with very long sentence."""
        # Single very long sentence
        long_sentence = "word " * 100  # 500 characters
        
        chunks = text_processor.chunk_text(long_sentence, max_chunk_size=50)
        
        assert len(chunks) > 1
        # Should split by words when sentence is too long
        for chunk in chunks:
            assert len(chunk) <= 50
    
    def test_process_complete_pipeline(self, text_processor, sample_texts):
        """Test complete text processing pipeline."""
        for text_type, text in sample_texts.items():
            if text_type in ["empty", "whitespace"]:
                # These should fail validation
                result = text_processor.process(text)
                assert result.is_valid is False
            else:
                result = text_processor.process(text)
                
                # Check result structure
                assert isinstance(result, TextProcessingResult)
                assert result.original_text == text
                assert isinstance(result.processed_text, str)
                assert isinstance(result.language, str)
                assert isinstance(result.word_count, int)
                assert isinstance(result.character_count, int)
                assert isinstance(result.warnings, list)
                
                if text_type != "empty":
                    assert result.is_valid is True
                    assert len(result.processed_text) > 0
                    assert result.word_count >= 0
                    assert result.character_count >= 0
    
    def test_process_with_warnings(self, text_processor):
        """Test processing that generates warnings."""
        # Long text should generate warning
        long_text = "word " * 600  # Should trigger word count warning
        result = text_processor.process(long_text)
        
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any("Long text" in warning for warning in result.warnings)
    
    def test_convert_two_digit_numbers(self, text_processor):
        """Test two-digit number conversion."""
        # Test the private method
        assert text_processor._convert_two_digit_number(25) == "twenty five"
        assert text_processor._convert_two_digit_number(30) == "thirty"
        assert text_processor._convert_two_digit_number(99) == "ninety nine"
    
    @pytest.mark.parametrize("input_text,expected_language", [
        ("Hello world", "en"),
        ("Bonjour le monde", "fr"),
        ("Hola el mundo", "es"),
        ("Hallo die Welt", "de"),
        ("", "en"),  # Default to English for empty text
    ])
    def test_language_detection_parametrized(self, text_processor, input_text, expected_language):
        """Parametrized test for language detection."""
        result = text_processor.detect_language(input_text)
        assert result == expected_language
    
    def test_regex_patterns_compilation(self, text_processor):
        """Test that regex patterns are properly compiled."""
        # Verify patterns exist and are compiled
        assert hasattr(text_processor, 'multiple_spaces_pattern')
        assert hasattr(text_processor, 'excessive_punct_pattern')
        assert hasattr(text_processor, 'number_pattern')
        assert hasattr(text_processor, 'ssml_pattern')
        assert hasattr(text_processor, 'email_pattern')
        assert hasattr(text_processor, 'url_pattern')
        
        # Test pattern functionality
        assert text_processor.multiple_spaces_pattern.sub(' ', 'a  b') == 'a b'
        assert text_processor.excessive_punct_pattern.sub('...', 'hello!!!!') == 'hello...'


class TestTextProcessorEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_non_string_input(self, text_processor):
        """Test handling of non-string input."""
        # Numbers
        result = text_processor.normalize(123)
        assert isinstance(result, str)
        
        # None
        result = text_processor.normalize(None)
        assert result == ""
        
        # Lists (should convert to string)
        result = text_processor.normalize(["hello", "world"])
        assert isinstance(result, str)
    
    def test_unicode_handling(self, text_processor):
        """Test Unicode text handling."""
        unicode_text = "Hello 世界 🌍"
        result = text_processor.process(unicode_text)
        
        assert result.is_valid is True
        assert "世界" in result.processed_text
        assert "🌍" in result.processed_text
    
    def test_extreme_length_text(self, text_processor):
        """Test handling of extremely long text."""
        # Create text at the boundary
        boundary_text = "x" * text_processor.max_length
        result = text_processor.validate(boundary_text)
        assert result is True
        
        # Just over the boundary
        over_boundary = "x" * (text_processor.max_length + 1)
        result = text_processor.validate(over_boundary)
        assert result is False
    
    def test_special_characters(self, text_processor):
        """Test handling of special characters."""
        special_text = "Hello\n\t\r\x00world"
        result = text_processor.normalize(special_text)
        
        # Should handle special characters gracefully
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_empty_chunk_text(self, text_processor):
        """Test chunking empty or whitespace text."""
        assert text_processor.chunk_text("") == []
        assert text_processor.chunk_text("   ") == []
        assert text_processor.chunk_text("\n\t") == []


@pytest.mark.slow
class TestTextProcessorPerformance:
    """Performance tests for TextProcessor."""
    
    def test_large_text_processing_performance(self, text_processor, performance_timer):
        """Test processing performance with large text."""
        # Create large text (10KB)
        large_text = "This is a test sentence. " * 400
        
        performance_timer.start()
        result = text_processor.process(large_text)
        elapsed = performance_timer.stop()
        
        assert result.is_valid is True
        assert elapsed < 1.0  # Should process 10KB in under 1 second
    
    def test_chunking_performance(self, text_processor, performance_timer):
        """Test chunking performance with very large text."""
        # Create very large text (100KB)
        very_large_text = "Word " * 20000
        
        performance_timer.start()
        chunks = text_processor.chunk_text(very_large_text, max_chunk_size=1000)
        elapsed = performance_timer.stop()
        
        assert len(chunks) > 0
        assert elapsed < 2.0  # Should chunk 100KB in under 2 seconds