from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict, Any
import httpx
import json

from app.config import settings


class LLMService(ABC):
    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str, history: List[Dict] = None) -> str:
        pass

    @abstractmethod
    async def generate_stream(self, prompt: str, system_prompt: str, history: List[Dict] = None) -> AsyncGenerator[str, None]:
        pass


class GroqService(LLMService):
    def __init__(self):
        from groq import AsyncGroq
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL

    def _build_messages(self, prompt: str, system_prompt: str, history: List[Dict] = None) -> List[Dict]:
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages

    async def test_connection(self) -> str:
        """Test the connection to the GROQ API."""
        try:
            response = await self.client.models.list()
            return f"Connection successful. Available models: {[model.id for model in response]}"
        except Exception as e:
            raise RuntimeError(f"Failed to connect to GROQ API: {str(e)}")

    async def generate(self, prompt: str, system_prompt: str, history: List[Dict] = None) -> str:
        messages = self._build_messages(prompt, system_prompt, history)
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_message = f"Groq API error: {str(e)}"
            print(error_message)  # Log the error for debugging
            raise RuntimeError(error_message)

    async def generate_stream(self, prompt: str, system_prompt: str, history: List[Dict] = None) -> AsyncGenerator[str, None]:
        messages = self._build_messages(prompt, system_prompt, history)
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1024,
                stream=True,
            )
            async for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    yield token
        except Exception as e:
            error_message = f"[ERROR] Groq error: {str(e)}"
            print(error_message)  # Log the error for debugging
            yield error_message


class OllamaService(LLMService):
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL

    def _build_messages(self, prompt: str, system_prompt: str, history: List[Dict] = None) -> List[Dict]:
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages

    async def generate(self, prompt: str, system_prompt: str, history: List[Dict] = None) -> str:
        messages = self._build_messages(prompt, system_prompt, history)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data["message"]["content"]
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except httpx.TimeoutException:
            raise TimeoutError("Ollama request timed out. The model may still be loading.")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama API error: {e.response.status_code} - {e.response.text}")

    async def generate_stream(self, prompt: str, system_prompt: str, history: List[Dict] = None) -> AsyncGenerator[str, None]:
        messages = self._build_messages(prompt, system_prompt, history)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                token = data.get("message", {}).get("content", "")
                                if token:
                                    yield token
                                if data.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue
        except httpx.ConnectError:
            yield "[ERROR] Cannot connect to Ollama. Make sure Ollama is running: `ollama serve`"
        except httpx.TimeoutException:
            yield "[ERROR] Ollama request timed out."
        except httpx.HTTPStatusError as e:
            yield f"[ERROR] Ollama returned {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            yield f"[ERROR] Unexpected error: {str(e)}"

    async def is_connected(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    async def is_model_loaded(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    return any(m.get("name", "").startswith(self.model.split(":")[0]) for m in models)
        except Exception:
            pass
        return False


def get_llm_service() -> LLMService:
    """Factory function — uses Groq if API key is set, otherwise falls back to Ollama."""
    if settings.GROQ_API_KEY:
        return GroqService()
    return OllamaService()
