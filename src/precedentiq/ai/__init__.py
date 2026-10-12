from .ollama_client import OllamaClient
from .holding_extractor import HoldingExtractor
from .argument_synthesizer import ArgumentSynthesizer
from .fallbacks import HeuristicLegalSynthesizer

__all__ = [
    "OllamaClient",
    "HoldingExtractor",
    "ArgumentSynthesizer",
    "HeuristicLegalSynthesizer",
]
