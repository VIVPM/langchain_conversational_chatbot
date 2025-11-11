import re
from typing import Tuple, Dict, Set
from detoxify import Detoxify

class InputGuardrails:
    """Guardrails for validating user input before LLM processing"""

    # Blocked phrases for prompt injection
    BLOCKED_PHRASES = [
        "ignore previous instructions", "ignore all previous", "you are now",
        "new instructions", "system:", "assistant:", "forget everything",
        "disregard", "jailbreak", "DAN mode", "pretend you are", "roleplay as"
    ]

    # PII patterns
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    PHONE_PATTERN = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    SSN_PATTERN = r'\b\d{3}-\d{2}-\d{4}\b'
    CREDIT_CARD_PATTERN = r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'

    # === New: explicit-content heuristics ===
    # Conservative, non-graphic patterns. Keep lowercase comparisons.
    EXPLICIT_KEYWORDS = {
        # general sexual content
        "porn", "nsfw", "xxx", "erotic", "onlyfans",
        # acts (kept generic)
        "sex", "sexual", "nude", "nudes", "naked", "strip", "explicit",
        # commercial
        "camgirl", "cam boy", "webcam show", "adult video",
    }

    # High-risk categories (separate so you can log/count distinctly)
    EXPLICIT_REGEXES = {
        "sexual_minors": re.compile(r"\b(child|minor|under\s*age|underage|teen\s*(?:\d{1,2})?)\b.*\b(sex|nude|explicit)\b", re.I),
        "incest": re.compile(r"\b(incest|step\s*(?:mom|mother|dad|father|sister|brother))\b", re.I),
        "bestiality": re.compile(r"\b(bestiality|zoophilia)\b", re.I),
        "fetish": re.compile(r"\b(fetish|bdsm)\b", re.I),
    }

    ADULT_DOMAINS = (
        "pornhub", "xvideos", "xhamster", "xnxx", "onlyfans", "redtube", "brazzers"
    )

    # Lazy-loaded toxicity model
    _toxicity_model = None

    @classmethod
    def get_toxicity_model(cls):
        if cls._toxicity_model is None:
            cls._toxicity_model = Detoxify("original")
        return cls._toxicity_model

    @staticmethod
    def validate_input(query: str) -> Tuple[bool, str]:
        if not query or not query.strip():
            return False, "Please enter a valid question."
        if len(query.strip()) < 3:
            return False, "Question is too short. Please be more specific."
        if len(query) > 2000:
            return False, "Question is too long. Please keep it under 2000 characters."

        ql = query.lower()

        # Prompt-injection phrases
        for phrase in InputGuardrails.BLOCKED_PHRASES:
            if phrase in ql:
                return False, "Your question contains prohibited content. Please rephrase."

        # PII
        if re.search(InputGuardrails.EMAIL_PATTERN, query):
            return False, "Please don't include email addresses in your question."
        if re.search(InputGuardrails.PHONE_PATTERN, query):
            return False, "Please don't include phone numbers in your question."
        if re.search(InputGuardrails.SSN_PATTERN, query):
            return False, "Please don't include sensitive ID numbers in your question."
        if re.search(InputGuardrails.CREDIT_CARD_PATTERN, query):
            return False, "Please don't include credit card numbers in your question."

        # Repetition
        if len(query) > 20:
            unique_chars = len(set(ql.replace(" ", "")))
            total_chars = len(query.replace(" ", ""))
            if total_chars > 0 and unique_chars < total_chars * 0.3:
                return False, "Your question appears to contain excessive repetition."

        # Special characters
        special_char_count = sum(1 for c in query if not c.isalnum() and not c.isspace())
        if special_char_count > len(query) * 0.5:
            return False, "Your question contains too many special characters."

        # Toxicity + explicit checks
        is_safe, msg = InputGuardrails.check_toxicity_and_explicit(query)
        if not is_safe:
            return False, msg

        return True, ""

    # === New: merged toxicity + explicit logic with per-label thresholds ===
    @staticmethod
    def check_toxicity_and_explicit(query: str) -> Tuple[bool, str]:
        ql = query.lower()

        # Per-label thresholds. Lower for sexual_explicit to catch more cases.
        thresholds: Dict[str, float] = {
            "toxicity": 0.50,
            "severe_toxicity": 0.50,
            "obscene": 0.50,
            "threat": 0.50,
            "insult": 0.50,
            "identity_attack": 0.50,
            "sexual_explicit": 0.35,       # catch more explicit content
            "sexually_explicit": 0.35,     # alternate key in some builds
        }

        # Run Detoxify, but handle absence/errors safely
        model_scores: Dict[str, float] = {}
        try:
            model = InputGuardrails.get_toxicity_model()
            model_scores = model.predict(query) or {}
        except Exception:
            # Fall back to heuristic-only block so lapses are visible
            model_scores = {}

        # Labels tripping threshold
        tripped = []
        for k, t in thresholds.items():
            if model_scores.get(k, 0.0) > t:
                tripped.append(k)

        # Heuristic explicit detection
        explicit_hits: Set[str] = set()

        # 1) Adult domains
        if any(dom in ql for dom in InputGuardrails.ADULT_DOMAINS):
            explicit_hits.add("pornography")

        # 2) Generic explicit keywords
        if any(kw in ql for kw in InputGuardrails.EXPLICIT_KEYWORDS):
            explicit_hits.add("sexual_explicit")

        # 3) High-risk regexes
        for label, rgx in InputGuardrails.EXPLICIT_REGEXES.items():
            if rgx.search(query):
                explicit_hits.add(label)

        # Combine
        if tripped or explicit_hits:
            # Friendly names
            label_map = {
                "sexual_explicit": "sexually explicit",
                "sexually_explicit": "sexually explicit",
                "identity_attack": "identity-based attacks",
                "severe_toxicity": "severe toxicity",
                "pornography": "pornographic content",
                "sexual_minors": "sexual content involving minors",
                "fetish": "fetish content",
            }
            human_labels = [label_map.get(k, k.replace("_", " ")) for k in (tripped + sorted(explicit_hits))]
            msg = f"Your question contains inappropriate content ({', '.join(human_labels)}). Please rephrase respectfully."
            return False, msg

        return True, ""
