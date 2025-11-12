import re
from typing import Tuple, Dict, Set
from detoxify import Detoxify

class InputGuardrails:
    """Guardrails for validating user input before LLM processing"""

    # Lazy-loaded toxicity model
    _toxicity_model = None

    @classmethod
    def get_toxicity_model(cls):
        if cls._toxicity_model is None:
            cls._toxicity_model = Detoxify("original")
        return cls._toxicity_model

    @staticmethod
    def check_toxicity_and_explicit(query: str) -> Tuple[bool, str]:
        cutoff = 0.05
        try:
            model = InputGuardrails.get_toxicity_model()
            scores: Dict[str, float] = model.predict(query) or {}
        except Exception:
            scores = {}

        tripped = [k for k, v in scores.items() if v > cutoff]

        if tripped:
            human_labels = [k.replace("_", " ") for k in tripped]
            return False, f"Your question contains inappropriate content ({', '.join(human_labels)}). Please rephrase respectfully."

        return True, ""

