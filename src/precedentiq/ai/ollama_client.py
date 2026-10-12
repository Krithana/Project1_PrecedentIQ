import json
import logging
from typing import Dict, Any, Optional
import httpx
from ..config import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for local Ollama LLM execution with JSON schema enforcement."""

    _missing_model_warnings = set()

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout or settings.OLLAMA_TIMEOUT_SECONDS

    def is_available(self) -> bool:
        """Check if Ollama server is running and responsive."""
        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    def is_model_available(self) -> bool:
        """Return true only when the configured model is exposed by this server."""
        models = self.list_models()
        available = {str(name) for name in models if name}
        if self.model in available:
            return True
        warning_key = f"{self.base_url}:{self.model}"
        if warning_key not in self._missing_model_warnings:
            logger.warning(
                "Ollama is reachable, but model '%s' is not available at %s. "
                "Using deterministic fallback.", self.model, self.base_url
            )
            self._missing_model_warnings.add(warning_key)
        return False

    def list_models(self) -> list:
        """List locally available models in Ollama."""
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                if resp.status_code == 200:
                    return [m.get("name") for m in resp.json().get("models", [])]
        except Exception:
            pass
        return []

    def generate_json(self, prompt: str, system: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Request structured JSON generation from Ollama.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,  # Low temperature for factual precision
                "top_p": 0.9,
            }
        }
        if system:
            payload["system"] = system

        # Avoid repeated HTTP 404 requests when the server is running without
        # the configured model. Callers can then use the deterministic engine.
        if not self.is_model_available():
            return None

        try:
            logger.info(f"Dispatching generation to Ollama model {self.model}...")
            with httpx.Client(timeout=float(self.timeout)) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 200:
                    result = resp.json()
                    response_text = result.get("response", "").strip()
                    try:
                        return json.loads(response_text)
                    except json.JSONDecodeError:
                        # Attempt to extract json object if wrapped in markdown
                        import re
                        match = re.search(r"\{.*\}", response_text, re.DOTALL)
                        if match:
                            return json.loads(match.group(0))
                        logger.warning(f"Failed to parse JSON response: {response_text[:200]}")
                        return None
                else:
                    logger.warning(f"Ollama returned HTTP {resp.status_code}: {resp.text}")
                    return None
        except httpx.TimeoutException:
            logger.warning(f"Ollama request timed out after {self.timeout}s.")
            return None
        except Exception as e:
            logger.warning(f"Ollama request failed: {e}")
            return None
