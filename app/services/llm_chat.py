# 1. dynamic user_prompt and model_type 
# 2. Rate Limit track and fallback with Retry logic
# 3. streaming Response 

from litellm import acompletion, stream_chunk_builder
import litellm
from litellm.exceptions import RateLimitError, APIError
from app.prompts.system_prompt import SYSTEM_PROMPT
import random
import time
import asyncio
import logging
from app.core.config import settings 
from app.ai.llm_metrics import metrics

logger = logging.getLogger(__name__)



MODEL_REGISTRY = {
    "fast": "gemini/gemini-2.5-flash",
    "pro": "groq/llama-3.1-8b-instant",
}

DEFAULT_MODEL_TYPE = "pro"


def resolve_model_name(model_type: str) -> str:
    """Maps a user-facing alias like 'fast' to the actual LiteLLM model string."""
    return MODEL_REGISTRY.get(model_type.lower().strip(), MODEL_REGISTRY[DEFAULT_MODEL_TYPE])



class ChatService:

    def __init__(self, model_type: str, user_prompt: str):

            self.user_prompt = user_prompt

            self.model_key = resolve_model_name(model_type)

            self.gemini_key = settings.gemini_api_key
            self.groq_key = settings.groq_api_key

            self.fallback_chain = [
                ("gemini/gemini-2.5-flash", self.gemini_key),
                ("groq/llama-3.1-8b-instant", self.groq_key),
            ]

            self.messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self.user_prompt},
            ]

    async def stream_llm_response(
        self,
        model: str,
        messages: list,
        max_retries: int = 3,
        **kwargs

    ):
        last_error = None

        for attempt in range(max_retries):

            try:
                response = await acompletion(
                    model = model,
                    messages = messages,   
                    api_key = kwargs['api_key'],
                    temperature = kwargs['temperature'],
                    max_tokens = kwargs['max_tokens'],
                    top_p = kwargs['top_p'],
                    stream = True
                )
                
                collected_chunks_list = []

                try:
                    async for chunk in response:
                        collected_chunks_list.append(chunk)

                        choices = getattr(chunk, "choices", [])
                        if choices:
                            delta_content = getattr(choices[0].delta, "content", "")
                            if delta_content:
                                yield delta_content

                    return

                except Exception as stream_err:
                    # If the connection drops or fails midway through the stream, yield a descriptive error token.
                    # This informs the client of the failure without breaking the connection protocol.
                    yield f"\n[ERROR: Stream interrupted - {str(stream_err)}]"

                finally:
                    # The finally block runs regardless of whether the stream succeeded or encountered an error.
                    # We use this opportunity to reconstruct the full response and log metrics.
                    if collected_chunks_list:
                        try:
                            # Reconstruct a standard non-streaming response object using LiteLLM utility
                            complete_response_object = stream_chunk_builder(collected_chunks_list)
                            
                            # Pass the reconstructed response to our existing metrics logger service
                            metrics.log_metrics(model_name=model, response=complete_response_object)
                        except Exception as metric_err:
                            print(f"Warning: Failed to log metrics for stream: {metric_err}")


            except RateLimitError as e:
                last_error = e
                logger.warning(f"[{model}] Rate limited (429) currently retrying ...({attempt}/{max_retries})")

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
                await asyncio.sleep(wait)    

            except APIError as e:
                last_error = e
                logger.error(f"[{model}] non retryable api error..({e})")
                raise e  


    async def chat(self):
        for model_name, api_key in self.fallback_chain:    
            try:
                async for chunk in self.stream_llm_response(   
                    model=self.model_key,
                    messages=self.messages,
                    api_key=api_key,
                    temperature=0.7,
                    max_tokens=500,
                    top_p=0.9,
                ):
                    yield chunk   # re-yield each chunk to the caller
                return            # if we reach here, streaming succeeded — stop fallback

            except Exception as e:
                logger.error(f"Model {model_name} failed, falling back... ({e})")
            


            
