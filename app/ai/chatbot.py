import logging
import json
import time
import random
from litellm import completion
from litellm.exceptions import RateLimitError, APIError
from app.prompts.system_prompt import SYSTEM_PROMPT
from app.ai.llm_metrics import metrics
from app.core.config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# STEP 1: Retry wrapper — handles one model
# ─────────────────────────────────────────────
def call_llm_with_retry(
    model: str,
    messages: list,
    api_key: str = None,
    max_retries: int = 3,
    **kwargs
):
    """
    Tries one model up to max_retries times.
    On 429 (rate limit), waits then retries.
    On final failure, raises the exception so the caller can fallback.
    """
    last_error = None  # ✅ Fix: always defined before the loop

    for attempt in range(max_retries): #attempts = 
        try:
            call_kwargs = {**kwargs}
            if api_key:
                call_kwargs["api_key"] = api_key

            response = completion(model=model, messages=messages, **call_kwargs)
            return response  # ✅ Success — exit immediately

        except RateLimitError as e:
            last_error = e

            if attempt == max_retries - 1: #attempt = 2
                logger.error(f"[{model}] Rate limit retries exhausted.")
                raise last_error  # ✅ Bubble up so ChatService can fallback

            # ✅ Fix: guard against headers being None
            headers = getattr(e, "headers", None) or {}
            retry_after_header = headers.get("Retry-After") or headers.get("retry-after")

            if retry_after_header:
                try:
                    wait = float(retry_after_header)
                except ValueError:
                    wait = (2 ** attempt) + random.uniform(0.1, 1.0)
            else:
                # Exponential backoff: 2s, 4s, 8s... + jitter
                wait = (2 ** attempt) + random.uniform(0.1, 1.0)

            logger.warning(
                f"[{model}] Rate limited (429). "
                f"Waiting {wait:.2f}s before retry {attempt + 1}/{max_retries}..."
            )
            time.sleep(wait)

        except APIError as e:
            # Non-rate-limit API error — no point retrying, raise immediately
            logger.error(f"[{model}] API error (non-retryable): {e}")
            raise e


# ─────────────────────────────────────────────
# STEP 2: Model registry — clean and extensible
# ─────────────────────────────────────────────

# Add or swap models here without touching any logic below
MODEL_REGISTRY = {
    "fast": "gemini/gemini-2.5-flash",
    "pro": "groq/llama-3.1-8b-instant",
}

DEFAULT_MODEL_KEY = "pro"


def resolve_model_name(model_key: str) -> str:
    """Maps a user-facing alias like 'fast' to the actual LiteLLM model string."""
    return MODEL_REGISTRY.get(model_key.lower().strip(), MODEL_REGISTRY[DEFAULT_MODEL_KEY])


# ─────────────────────────────────────────────
# STEP 3: ChatService — orchestrates fallback
# ─────────────────────────────────────────────

class ChatService:

    @staticmethod
    def chat_model(
        model_key: str,
        user_prompt: str,
        system_prompt: str = SYSTEM_PROMPT
    ) -> dict:
        """
        1. Resolves 'fast'/'smart' → real model name
        2. Tries primary model with retries
        3. Falls back to next model in chain if it fails
        4. Returns structured result dict
        """
        gemini_key = settings.gemini_api_key
        groq_key = settings.groq_api_key

        # ✅ Fix: model_key is preserved, model_name is a separate resolved variable
        primary_model = resolve_model_name(model_key)

        # Fallback chain — always tried in order
        # Even if primary is gemini, gemini is tried first here
        fallback_chain = [
            ("gemini/gemini-2.5-flash", gemini_key),
            ("groq/llama-3.1-8b-instant", groq_key),
        ]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        last_error = None  # ✅ Fix: always defined

        for model_name, api_key in fallback_chain:
            if not api_key:
                logger.debug(f"Skipping {model_name} — API key missing.")
                continue

            try:
                logger.info(f"Trying model: {model_name}...")
                response = call_llm_with_retry(
                    model=model_name,
                    messages=messages,
                    api_key=api_key,
                    temperature=0.6,
                    max_tokens=500,
                    top_p=0.9,
                )

                metrics_result = metrics.log_metrics(model_name=model_name, response=response)

                return {
                    "success": True,
                    "model_used": model_name,
                    "response": response["choices"][0]["message"]["content"],
                    "model": metrics_result,
                }

            except Exception as e:
                last_error = e  # ✅ Always captured
                logger.warning(f"Model {model_name} failed, trying next... ({e})")
                continue  # Move to next in fallback chain

        # All models exhausted
        logger.error("All models in fallback chain failed.")
        return {
            "success": False,
            "response": "All available AI models failed. Please try again later.",
            "error": str(last_error),  # ✅ Now safe — always defined
        }


chat = ChatService()

if __name__ == "__main__":
    print(chat.chat_model("smart", "Hello, how are you?"))

