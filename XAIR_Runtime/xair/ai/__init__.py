"""AI-native intent producers for Paper 2 (VLM/LLM → AIS)."""

from xair.ai.ollama_client import OllamaClient
from xair.ai.structured_intent import StructuredIntentProducer

__all__ = ["OllamaClient", "StructuredIntentProducer"]
