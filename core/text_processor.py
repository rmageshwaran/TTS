"""
Text Processor Module

Handles text preprocessing and validation for TTS conversion.
Includes text cleaning, normalization, SSML processing, and language detection.
"""

import re
import logging
import unicodedata
import html
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from xml.etree import ElementTree as ET
from xml.parsers.expat import ExpatError

logger = logging.getLogger(__name__)


@dataclass
class TextProcessingResult:
    """Result of text processing operation."""
    original_text: str
    processed_text: str
    language: str
    word_count: int
    character_count: int
    is_valid: bool
    warnings: List[str]
    chunks: Optional[List[str]] = None
    ssml_tags: Optional[List[str]] = None
    pronunciation_notes: Optional[Dict[str, str]] = None


class TextProcessor:
    """
    Handles text preprocessing and validation for TTS conversion.
    
    Features:
    - Text cleaning and normalization
    - SSML support for speech control
    - Character limits and validation
    - Multi-language text detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TextProcessor with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        text_config = config.get("text", {})
        
        self.max_length = text_config.get("max_length", 5000)
        self.min_length = text_config.get("min_length", 1)
        self.enable_ssml = text_config.get("enable_ssml", True)
        self.default_language = text_config.get("default_language", "en")
        self.chunk_size = text_config.get("chunk_size", 1000)
        self.enable_pronunciation = text_config.get("enable_pronunciation", True)
        
        # Custom pronunciations dictionary
        self.custom_pronunciations: Dict[str, str] = text_config.get("custom_pronunciations", {})
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        # Initialize phoneme conversion (basic implementation)
        self._init_phoneme_converter()
        
        logger.info("TextProcessor initialized with enhanced features")
    
    def _init_phoneme_converter(self):
        """Initialize basic phoneme conversion system."""
        # Basic phoneme mappings (can be expanded)
        self.phoneme_map = {
            # Common word pronunciations
            "read": "reed",  # present tense
            "lead": "leed",  # verb form
            "live": "liv",   # verb form
            "wind": "wynd",  # verb form
            "tear": "tair",  # verb form
            "bow": "boh",    # verb form
            # Technical terms
            "API": "A P I",
            "URL": "U R L",
            "HTTP": "H T T P",
            "HTTPS": "H T T P S",
            "CPU": "C P U",
            "GPU": "G P U",
            "RAM": "R A M",
            "USB": "U S B",
            # Add more as needed
        }
        
        # Merge with custom pronunciations
        self.phoneme_map.update(self.custom_pronunciations)
    
    def _compile_patterns(self):
        """Compile regex patterns for text processing."""
        # Pattern to detect multiple spaces
        self.multiple_spaces_pattern = re.compile(r'\s+')
        
        # Pattern to detect excessive punctuation
        self.excessive_punct_pattern = re.compile(r'[.!?]{3,}')
        
        # Pattern to detect numbers (enhanced)
        self.number_pattern = re.compile(r'\b\d+(?:[.,]\d+)*\b')
        
        # Pattern to detect decimal numbers
        self.decimal_pattern = re.compile(r'\b\d+\.\d+\b')
        
        # Pattern to detect ordinal numbers
        self.ordinal_pattern = re.compile(r'\b\d+(?:st|nd|rd|th)\b', re.IGNORECASE)
        
        # Pattern to detect time formats
        self.time_pattern = re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:AM|PM))?\b', re.IGNORECASE)
        
        # Pattern to detect dates
        self.date_pattern = re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b')
        
        # Pattern to detect SSML tags (more comprehensive)
        self.ssml_pattern = re.compile(r'<[^>]+>', re.IGNORECASE)
        
        # Pattern to detect specific SSML tags
        self.ssml_speak_pattern = re.compile(r'<speak[^>]*>(.*?)</speak>', re.DOTALL | re.IGNORECASE)
        self.ssml_break_pattern = re.compile(r'<break[^>]*/?>', re.IGNORECASE)
        self.ssml_emphasis_pattern = re.compile(r'<emphasis[^>]*>(.*?)</emphasis>', re.DOTALL | re.IGNORECASE)
        
        # Pattern to detect email addresses
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # Pattern to detect URLs (enhanced)
        self.url_pattern = re.compile(r'https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        
        # Pattern to detect phone numbers
        self.phone_pattern = re.compile(r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b')
        
        # Pattern to detect currency
        self.currency_pattern = re.compile(r'\$\d+(?:[.,]\d+)*|\d+(?:[.,]\d+)*\s*(?:dollars?|cents?)', re.IGNORECASE)
        
        # Pattern to detect abbreviations
        self.abbreviation_pattern = re.compile(r'\b[A-Z]{2,}\b')
        
        # Pattern to detect contractions
        self.contraction_pattern = re.compile(r"\b\w+'\w+\b")
    
    def validate(self, text: str) -> Tuple[bool, List[str]]:
        """
        Validate input text with detailed error reporting.
        
        Args:
            text: Input text to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        if not text:
            errors.append("Text is empty or None")
            return False, errors
            
        if not isinstance(text, str):
            errors.append(f"Text must be string, got {type(text).__name__}")
            return False, errors
            
        # Check length constraints
        if len(text) < self.min_length:
            errors.append(f"Text too short (minimum {self.min_length} characters)")
            
        if len(text) > self.max_length:
            errors.append(f"Text too long (maximum {self.max_length} characters)")
            
        # Check for valid characters
        if not text.strip():
            errors.append("Text contains only whitespace")
            
        # Check for control characters
        if any(ord(char) < 32 and char not in '\t\n\r' for char in text):
            errors.append("Text contains invalid control characters")
            
        # Check encoding
        try:
            text.encode('utf-8')
        except UnicodeEncodeError:
            errors.append("Text contains invalid Unicode characters")
            
        # SSML validation if enabled
        if self.enable_ssml and '<' in text and '>' in text:
            ssml_errors = self._validate_ssml(text)
            errors.extend(ssml_errors)
            
        return len(errors) == 0, errors
    
    def _validate_ssml(self, text: str) -> List[str]:
        """
        Validate SSML syntax in text.
        
        Args:
            text: Text potentially containing SSML
            
        Returns:
            List[str]: List of SSML validation errors
        """
        errors = []
        
        try:
            # Try to parse as XML to check basic syntax
            # Wrap in root element if not already wrapped
            if not text.strip().startswith('<speak'):
                test_text = f'<speak>{text}</speak>'
            else:
                test_text = text
                
            # Basic XML parsing
            ET.fromstring(test_text)
            
        except (ExpatError, ET.ParseError) as e:
            errors.append(f"Invalid SSML syntax: {str(e)}")
        except Exception as e:
            errors.append(f"SSML validation error: {str(e)}")
            
        # Check for valid SSML tags (basic list)
        valid_ssml_tags = {
            'speak', 'break', 'emphasis', 'prosody', 'say-as', 'phoneme',
            'sub', 'audio', 'mark', 'voice', 'lang', 'p', 's'
        }
        
        # Extract tag names and validate
        tag_pattern = re.compile(r'<(/?)(\w+)(?:\s|>)')
        for match in tag_pattern.finditer(text):
            tag_name = match.group(2).lower()
            if tag_name not in valid_ssml_tags:
                errors.append(f"Unsupported SSML tag: {tag_name}")
                
        return errors
    
    def normalize(self, text: str) -> str:
        """
        Advanced text normalization for TTS processing.
        
        Args:
            text: Input text to normalize
            
        Returns:
            str: Normalized text
        """
        if not text:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # HTML decode (in case text contains HTML entities)
        text = html.unescape(text)
        
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Replace multiple spaces with single space
        text = self.multiple_spaces_pattern.sub(' ', text)
        
        # Normalize excessive punctuation
        text = self.excessive_punct_pattern.sub('...', text)
        
        # Process different text elements in order
        text = self._normalize_currency(text)
        text = self._normalize_phone_numbers(text)
        text = self._normalize_dates(text)
        text = self._normalize_times(text)
        text = self._normalize_ordinals(text)
        text = self._normalize_decimals(text)
        text = self._convert_numbers_to_words(text)
        text = self._normalize_abbreviations(text)
        text = self._normalize_contractions(text)
        text = self._normalize_emails(text)
        text = self._normalize_urls(text)
        
        # Apply custom pronunciations
        if self.enable_pronunciation:
            text = self._apply_custom_pronunciations(text)
        
        # Final cleanup
        text = self.multiple_spaces_pattern.sub(' ', text)
        text = text.strip()
        
        return text
    
    def _convert_numbers_to_words(self, text: str) -> str:
        """
        Convert numbers to words for better TTS pronunciation.
        
        Args:
            text: Input text containing numbers
            
        Returns:
            str: Text with numbers converted to words
        """
        # Basic number conversion (extend as needed)
        number_words = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
            '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
            '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
            '18': 'eighteen', '19': 'nineteen', '20': 'twenty'
        }
        
        def replace_number(match):
            number = match.group()
            if number in number_words:
                return number_words[number]
            elif len(number) <= 2:
                # Handle two-digit numbers
                if int(number) < 100:
                    return self._convert_two_digit_number(int(number))
            return number  # Keep original if conversion is complex
        
        return self.number_pattern.sub(replace_number, text)
    
    def _convert_two_digit_number(self, num: int) -> str:
        """Convert two-digit number to words."""
        if num < 20:
            return str(num)  # Will be handled by basic conversion
        
        tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
        ones = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        
        tens_digit = num // 10
        ones_digit = num % 10
        
        result = tens[tens_digit]
        if ones_digit > 0:
            result += ' ' + ones[ones_digit] if result else ones[ones_digit]
        
        return result
    
    def _normalize_emails(self, text: str) -> str:
        """Replace email addresses with spoken equivalent."""
        def replace_email(match):
            email = match.group()
            # Simple replacement - can be enhanced
            return email.replace('@', ' at ').replace('.', ' dot ')
        
        return self.email_pattern.sub(replace_email, text)
    
    def _normalize_urls(self, text: str) -> str:
        """Replace URLs with spoken equivalent."""
        return self.url_pattern.sub('URL', text)
    
    def _normalize_currency(self, text: str) -> str:
        """Normalize currency expressions."""
        def replace_currency(match):
            amount = match.group()
            if amount.startswith('$'):
                return amount[1:] + ' dollars'
            return amount
        
        return self.currency_pattern.sub(replace_currency, text)
    
    def _normalize_phone_numbers(self, text: str) -> str:
        """Normalize phone numbers to spoken format."""
        def replace_phone(match):
            phone = match.group()
            # Remove formatting and convert to spoken digits
            digits = re.sub(r'[^\d]', '', phone)
            if len(digits) == 10:
                return f"{digits[:3]} {digits[3:6]} {digits[6:]}"
            elif len(digits) == 11 and digits[0] == '1':
                return f"1 {digits[1:4]} {digits[4:7]} {digits[7:]}"
            return phone
        
        return self.phone_pattern.sub(replace_phone, text)
    
    def _normalize_dates(self, text: str) -> str:
        """Normalize date formats to spoken form."""
        def replace_date(match):
            date_str = match.group()
            # Basic date normalization (can be enhanced)
            return date_str.replace('/', ' ').replace('-', ' ')
        
        return self.date_pattern.sub(replace_date, text)
    
    def _normalize_times(self, text: str) -> str:
        """Normalize time formats to spoken form."""
        def replace_time(match):
            time_str = match.group()
            # Basic time normalization
            if 'AM' in time_str.upper() or 'PM' in time_str.upper():
                return time_str.replace(':', ' ')
            return time_str.replace(':', ' ')
        
        return self.time_pattern.sub(replace_time, text)
    
    def _normalize_ordinals(self, text: str) -> str:
        """Normalize ordinal numbers."""
        def replace_ordinal(match):
            ordinal = match.group()
            number = re.search(r'\d+', ordinal).group()
            suffix = ordinal[len(number):].lower()
            
            # Convert number to word and add ordinal suffix
            if int(number) <= 20:
                word_map = {
                    '1': 'first', '2': 'second', '3': 'third', '4': 'fourth', '5': 'fifth',
                    '6': 'sixth', '7': 'seventh', '8': 'eighth', '9': 'ninth', '10': 'tenth',
                    '11': 'eleventh', '12': 'twelfth', '13': 'thirteenth', '14': 'fourteenth',
                    '15': 'fifteenth', '16': 'sixteenth', '17': 'seventeenth', '18': 'eighteenth',
                    '19': 'nineteenth', '20': 'twentieth'
                }
                return word_map.get(number, ordinal)
            
            return ordinal  # Keep as-is for complex ordinals
        
        return self.ordinal_pattern.sub(replace_ordinal, text)
    
    def _normalize_decimals(self, text: str) -> str:
        """Normalize decimal numbers."""
        def replace_decimal(match):
            decimal = match.group()
            parts = decimal.split('.')
            return f"{parts[0]} point {' '.join(parts[1])}"
        
        return self.decimal_pattern.sub(replace_decimal, text)
    
    def _normalize_abbreviations(self, text: str) -> str:
        """Normalize common abbreviations."""
        abbreviations = {
            'Mr.': 'Mister',
            'Mrs.': 'Missus',
            'Dr.': 'Doctor',
            'Prof.': 'Professor',
            'Inc.': 'Incorporated',
            'Ltd.': 'Limited',
            'Corp.': 'Corporation',
            'Co.': 'Company',
            'etc.': 'etcetera',
            'vs.': 'versus',
            'e.g.': 'for example',
            'i.e.': 'that is'
        }
        
        for abbrev, expansion in abbreviations.items():
            text = text.replace(abbrev, expansion)
        
        # Handle letter-by-letter abbreviations
        def replace_abbrev(match):
            abbrev = match.group()
            if len(abbrev) <= 5:  # Short abbreviations
                return ' '.join(abbrev)
            return abbrev
        
        return self.abbreviation_pattern.sub(replace_abbrev, text)
    
    def _normalize_contractions(self, text: str) -> str:
        """Expand contractions."""
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
            "'s": " is"  # Note: this is ambiguous (is/has)
        }
        
        for contraction, expansion in contractions.items():
            text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def _apply_custom_pronunciations(self, text: str) -> str:
        """Apply custom pronunciation mappings."""
        words = text.split()
        result = []
        
        for word in words:
            # Clean word for lookup (remove punctuation)
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Check for exact match first
            if clean_word in self.phoneme_map:
                # Preserve punctuation
                pronunciation = self.phoneme_map[clean_word]
                punctuation = re.sub(r'\w', '', word)
                result.append(pronunciation + punctuation)
            elif clean_word.upper() in self.phoneme_map:
                pronunciation = self.phoneme_map[clean_word.upper()]
                punctuation = re.sub(r'\w', '', word)
                result.append(pronunciation + punctuation)
            else:
                result.append(word)
        
        return ' '.join(result)
    
    def process_ssml(self, text: str) -> Tuple[str, List[str]]:
        """
        Advanced SSML processing with tag extraction and validation.
        
        Args:
            text: Input text potentially containing SSML
            
        Returns:
            Tuple[str, List[str]]: (processed_text, extracted_ssml_tags)
        """
        extracted_tags = []
        
        if not self.enable_ssml:
            # Remove all SSML tags if disabled
            cleaned_text = self.ssml_pattern.sub('', text)
            return cleaned_text, []
        
        # Extract SSML tags for analysis
        for match in self.ssml_pattern.finditer(text):
            extracted_tags.append(match.group())
        
        # Validate SSML if present
        if extracted_tags:
            errors = self._validate_ssml(text)
            if errors:
                logger.warning(f"SSML validation errors: {errors}")
        
        # Process specific SSML constructs
        processed_text = self._process_ssml_breaks(text)
        processed_text = self._process_ssml_emphasis(processed_text)
        
        return processed_text, extracted_tags
    
    def _process_ssml_breaks(self, text: str) -> str:
        """Process SSML break tags."""
        # Convert break tags to natural pauses
        def replace_break(match):
            break_tag = match.group()
            if 'time=' in break_tag:
                # Extract time value (basic implementation)
                return ' ... '  # Represent as ellipsis
            return ' '  # Short pause
        
        return self.ssml_break_pattern.sub(replace_break, text)
    
    def _process_ssml_emphasis(self, text: str) -> str:
        """Process SSML emphasis tags."""
        # For now, just remove the tags but keep content
        def replace_emphasis(match):
            content = match.group(1)
            return content.upper()  # Simple emphasis representation
        
        return self.ssml_emphasis_pattern.sub(replace_emphasis, text)
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Enhanced language detection with confidence scoring.
        
        Args:
            text: Input text
            
        Returns:
            Tuple[str, float]: (language_code, confidence_score)
        """
        if not text or not text.strip():
            return self.default_language, 0.0
        
        text_lower = text.lower()
        word_count = len(text.split())
        
        # Language patterns with weights
        language_patterns = {
            'fr': {
                'patterns': [r'\b(le|la|les|un|une|des|et|ou|de|du|que|qui|avec|dans|pour|sur|par|est|sont|avoir|être)\b'],
                'weight': 1.0
            },
            'es': {
                'patterns': [r'\b(el|la|los|las|un|una|y|o|de|del|que|quien|con|en|para|por|es|son|tener|ser)\b'],
                'weight': 1.0
            },
            'de': {
                'patterns': [r'\b(der|die|das|ein|eine|und|oder|von|zu|dass|wer|mit|in|für|auf|durch|ist|sind|haben|sein)\b'],
                'weight': 1.0
            },
            'it': {
                'patterns': [r'\b(il|la|le|un|una|e|o|di|del|che|chi|con|in|per|su|da|è|sono|avere|essere)\b'],
                'weight': 1.0
            },
            'pt': {
                'patterns': [r'\b(o|a|os|as|um|uma|e|ou|de|do|que|quem|com|em|para|por|é|são|ter|ser)\b'],
                'weight': 1.0
            }
        }
        
        scores = {}
        
        for lang, config in language_patterns.items():
            score = 0
            for pattern in config['patterns']:
                matches = len(re.findall(pattern, text_lower))
                score += matches * config['weight']
            
            # Normalize by word count
            if word_count > 0:
                scores[lang] = score / word_count
            else:
                scores[lang] = 0
        
        # Find best match
        if scores:
            best_lang = max(scores, key=scores.get)
            confidence = scores[best_lang]
            
            # Require minimum confidence threshold
            if confidence >= 0.1:  # At least 10% of words match
                return best_lang, confidence
        
        # Default to configured language
        return self.default_language, 0.0
    
    def chunk_text(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """
        Split long text into chunks for processing.
        
        Args:
            text: Input text to chunk
            max_chunk_size: Maximum characters per chunk
            
        Returns:
            List[str]: List of text chunks
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        sentences = re.split(r'[.!?]+', text)
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Single sentence is too long, split by words
                    words = sentence.split()
                    word_chunk = ""
                    for word in words:
                        if len(word_chunk) + len(word) + 1 > max_chunk_size:
                            if word_chunk:
                                chunks.append(word_chunk.strip())
                                word_chunk = word
                            else:
                                # Single word is too long, truncate
                                chunks.append(word[:max_chunk_size])
                        else:
                            word_chunk += " " + word if word_chunk else word
                    if word_chunk:
                        current_chunk = word_chunk
            else:
                current_chunk += ". " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk.strip()]  # Filter empty chunks
    
    def process(self, text: str) -> TextProcessingResult:
        """
        Process text through the complete enhanced pipeline.
        
        Args:
            text: Input text to process
            
        Returns:
            TextProcessingResult: Complete processing result with enhanced features
        """
        warnings = []
        
        # Validate input with detailed error reporting
        is_valid, validation_errors = self.validate(text)
        if not is_valid:
            return TextProcessingResult(
                original_text=text,
                processed_text="",
                language="unknown",
                word_count=0,
                character_count=0,
                is_valid=False,
                warnings=validation_errors
            )
        
        # Process text through enhanced pipeline
        processed_text = self.normalize(text)
        processed_text, ssml_tags = self.process_ssml(processed_text)
        
        # Enhanced language detection with confidence
        language, language_confidence = self.detect_language(processed_text)
        if language_confidence < 0.3:
            warnings.append(f"Low confidence language detection: {language} ({language_confidence:.2f})")
        
        # Calculate statistics
        word_count = len(processed_text.split())
        character_count = len(processed_text)
        
        # Generate text chunks if needed
        chunks = None
        if character_count > self.chunk_size:
            chunks = self.chunk_text(processed_text, self.chunk_size)
            warnings.append(f"Text chunked into {len(chunks)} segments")
        
        # Advanced warnings
        if character_count > self.max_length * 0.8:
            warnings.append("Text is close to maximum length limit")
        
        if word_count > 500:
            warnings.append("Long text may take time to process")
        
        if len(ssml_tags) > 0:
            warnings.append(f"Found {len(ssml_tags)} SSML tags")
        
        # Check for potential pronunciation issues
        pronunciation_notes = self._analyze_pronunciation_challenges(processed_text)
        if pronunciation_notes:
            warnings.append(f"Found {len(pronunciation_notes)} potential pronunciation challenges")
        
        logger.debug(f"Enhanced processing: {character_count} chars, {word_count} words, language: {language} ({language_confidence:.2f})")
        
        return TextProcessingResult(
            original_text=text,
            processed_text=processed_text,
            language=language,
            word_count=word_count,
            character_count=character_count,
            is_valid=True,
            warnings=warnings,
            chunks=chunks,
            ssml_tags=ssml_tags,
            pronunciation_notes=pronunciation_notes
        )
    
    def _analyze_pronunciation_challenges(self, text: str) -> Dict[str, str]:
        """
        Analyze text for potential pronunciation challenges.
        
        Args:
            text: Processed text to analyze
            
        Returns:
            Dict[str, str]: Dictionary of challenging words and suggested pronunciations
        """
        challenges = {}
        
        # Common challenging words (can be expanded)
        challenging_patterns = {
            r'\brough\b': 'ruff',
            r'\bthrough\b': 'throo',
            r'\bcough\b': 'coff',
            r'\benough\b': 'enuff',
            r'\bcolonel\b': 'kernel',
            r'\breceipt\b': 'reseet',
            r'\bisland\b': 'eyeland',
            r'\bknee\b': 'nee',
            r'\bknow\b': 'noh',
            r'\bwrong\b': 'rong'
        }
        
        for pattern, pronunciation in challenging_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                word = match.group()
                challenges[word] = pronunciation
        
        return challenges
    
    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get comprehensive text statistics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict[str, Any]: Comprehensive text statistics
        """
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'average_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'average_sentence_length': len(words) / len(sentences) if sentences else 0,
            'language_complexity': self._calculate_complexity_score(text),
            'readability_score': self._calculate_readability_score(text)
        }
    
    def _calculate_complexity_score(self, text: str) -> float:
        """
        Calculate text complexity score (0-1, higher = more complex).
        
        Args:
            text: Text to analyze
            
        Returns:
            float: Complexity score
        """
        words = text.split()
        if not words:
            return 0.0
        
        # Factors that increase complexity
        long_words = sum(1 for word in words if len(word) > 6)
        rare_punctuation = len(re.findall(r'[;:()\[\]{}]', text))
        numbers = len(re.findall(r'\d+', text))
        
        complexity = (
            (long_words / len(words)) * 0.4 +
            (rare_punctuation / len(text)) * 100 * 0.3 +
            (numbers / len(words)) * 0.3
        )
        
        return min(complexity, 1.0)
    
    def _calculate_readability_score(self, text: str) -> float:
        """
        Calculate readability score (simplified Flesch Reading Ease).
        
        Args:
            text: Text to analyze
            
        Returns:
            float: Readability score (0-100, higher = more readable)
        """
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        if not words or not sentences:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables = self._estimate_syllables(text) / len(words)
        
        # Simplified Flesch formula
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        
        return max(0, min(100, score))
    
    def _estimate_syllables(self, text: str) -> int:
        """
        Estimate syllable count in text (basic implementation).
        
        Args:
            text: Text to analyze
            
        Returns:
            int: Estimated syllable count
        """
        # Basic syllable counting heuristic
        words = re.findall(r'\b\w+\b', text.lower())
        syllable_count = 0
        
        for word in words:
            # Count vowel groups
            vowels = 'aeiouy'
            prev_was_vowel = False
            word_syllables = 0
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    word_syllables += 1
                prev_was_vowel = is_vowel
            
            # Adjust for silent 'e'
            if word.endswith('e') and word_syllables > 1:
                word_syllables -= 1
            
            # Minimum one syllable per word
            syllable_count += max(1, word_syllables)
        
        return syllable_count