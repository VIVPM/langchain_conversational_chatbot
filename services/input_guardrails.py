import re
from typing import Tuple
from detoxify import Detoxify

class InputGuardrails:
    """Guardrails for validating user input before LLM processing"""
    
    # Blocked phrases for prompt injection
    BLOCKED_PHRASES = [
        "ignore previous instructions",
        "ignore all previous",
        "you are now",
        "new instructions",
        "system:",
        "assistant:",
        "forget everything",
        "disregard",
        "jailbreak",
        "DAN mode",
        "pretend you are",
        "roleplay as"
    ]
    
    # PII patterns
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    PHONE_PATTERN = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    SSN_PATTERN = r'\b\d{3}-\d{2}-\d{4}\b'
    CREDIT_CARD_PATTERN = r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
    
    # Initialize toxicity model (lazy loading)
    _toxicity_model = None
    
    @classmethod
    def get_toxicity_model(cls):
        """Lazy load the toxicity detection model"""
        if cls._toxicity_model is None:
            cls._toxicity_model = Detoxify('original')
        return cls._toxicity_model
    
    @staticmethod
    def validate_input(query: str) -> Tuple[bool, str]:
        """
        Validates user input through multiple guardrails.
        Returns (is_valid, error_message)
        """
        
        # 1. Length checks
        if not query or not query.strip():
            return False, "Please enter a valid question."
        
        if len(query.strip()) < 3:
            return False, "Question is too short. Please be more specific."
        
        if len(query) > 2000:
            return False, "Question is too long. Please keep it under 2000 characters."
        
        # 2. Prompt injection detection
        query_lower = query.lower()
        for phrase in InputGuardrails.BLOCKED_PHRASES:
            if phrase in query_lower:
                return False, "Your question contains prohibited content. Please rephrase."
        
        # 3. PII detection
        if re.search(InputGuardrails.EMAIL_PATTERN, query):
            return False, "Please don't include email addresses in your question."
        
        if re.search(InputGuardrails.PHONE_PATTERN, query):
            return False, "Please don't include phone numbers in your question."
        
        if re.search(InputGuardrails.SSN_PATTERN, query):
            return False, "Please don't include sensitive ID numbers in your question."
        
        if re.search(InputGuardrails.CREDIT_CARD_PATTERN, query):
            return False, "Please don't include credit card numbers in your question."
        
        # 4. Excessive repetition check
        if len(query) > 20:  # Only check for longer queries
            unique_chars = len(set(query.lower().replace(' ', '')))
            total_chars = len(query.replace(' ', ''))
            if total_chars > 0 and unique_chars < total_chars * 0.3:
                return False, "Your question appears to contain excessive repetition."
        
        # 5. Excessive special characters
        special_char_count = sum(1 for c in query if not c.isalnum() and not c.isspace())
        if special_char_count > len(query) * 0.5:
            return False, "Your question contains too many special characters."
        
        # 6. Toxicity detection using ML model
        is_safe, toxicity_msg = InputGuardrails.check_toxicity(query)
        if not is_safe:
            return False, toxicity_msg
        
        return True, ""
    
    @staticmethod
    def check_toxicity(query: str, threshold: float = 0.7) -> Tuple[bool, str]:
        """
        Check for toxic content using Detoxify ML model.
        Returns (is_safe, error_message)
        """
        try:
            model = InputGuardrails.get_toxicity_model()
            results = model.predict(query)
            
            # Check each toxicity category
            toxic_categories = []
            if results.get('toxicity', 0) > threshold:
                toxic_categories.append('toxic language')
            if results.get('severe_toxicity', 0) > threshold:
                toxic_categories.append('severe toxicity')
            if results.get('obscene', 0) > threshold:
                toxic_categories.append('obscene content')
            if results.get('threat', 0) > threshold:
                toxic_categories.append('threatening language')
            if results.get('insult', 0) > threshold:
                toxic_categories.append('insulting language')
            if results.get('identity_attack', 0) > threshold:
                toxic_categories.append('identity-based attacks')
            
            if toxic_categories:
                return False, f"Your question contains inappropriate content ({', '.join(toxic_categories)}). Please rephrase respectfully."
            
            return True, ""
        
        except Exception as e:
            # If toxicity check fails, log but don't block the user
            print(f"Toxicity check error: {e}")
            return True, ""
