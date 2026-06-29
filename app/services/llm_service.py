import asyncio
import litellm
from litellm import acompletion, stream_chunk_builder
from litellm.exceptions import RateLimitError, APIError, APIConnectionError, AuthenticationError
from fastapi import HTTPException
from app.services.metrics_service import MetricsTracker
from app.prompt.system_prompt import SYSTEM_PROMPT

# Enable returning response headers for rate limit tracking
litellm.return_response_headers = True


async def stream_generator(api_response_stream, model_name: str):
    """
    An async generator that processes the streaming response chunk by chunk.
    
    This function:
    1. Iterates asynchronously over chunks sent by the LLM provider.
    2. Extracts the text delta (the new word or characters) and yields it to the client.
    3. Collects all chunks in a list to reconstruct the complete message at the end.
    4. Automatically calculates final API metrics (token usage, cost) when finished.
    
    Args:
        api_response_stream: The async generator object returned by acompletion().
        model_name: The string name of the model being called (e.g. 'gemini/gemini-2.5-flash').
        
    Yields:
        str: Text content chunks of the LLM response.
    """
    collected_chunks_list = []
    
    try:
        # Asynchronously fetch chunks as they arrive from the API provider
        async for chunk in api_response_stream:
            collected_chunks_list.append(chunk)
            
            # Extract the new delta characters from the current chunk, if available
            choices = getattr(chunk, "choices", [])
            if choices:
                delta_content = getattr(choices[0].delta, "content", "")
                if delta_content:
                    # Yield the text chunk immediately to the FastAPI response stream
                    yield delta_content
                    
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
                MetricsTracker.log_metrics(response=complete_response_object, model_name=model_name)
            except Exception as metric_err:
                print(f"Warning: Failed to log metrics for stream: {metric_err}")


class ChatService:
    """
    Service responsible for managing LLM conversations.
    Handles mapping model aliases, setting up prompts, and initializing streams.
    """
    
    def __init__(self, model_type: str, user_prompt: str):
        """
        Initializes ChatService and resolves user-friendly model aliases to LiteLLM model identifiers.
        
        Args:
            model_type: The requested model alias ('fast', 'pro', 'slow') or a direct model string.
            user_prompt: The main user message prompt.
        """
        self.user_prompt = user_prompt
        self.system_prompt = SYSTEM_PROMPT
        
        # Standardize and map the user-friendly model nicknames to exact provider strings
        # We store the lowercased version to use consistently throughout this method
        model_lower = model_type.lower().strip()  # ✅ single source of truth for comparisons

        if model_lower == "fast":
            self.model = "gemini/gemini-2.5-flash"
            self.fallback_model = "groq/llama-3.3-70b-versatile"

        elif model_lower in ("pro", "slow"):
            self.model = "groq/llama-3.3-70b-versatile"
            self.fallback_model = "gemini/gemini-2.5-flash"

        else:
            # Allows power users to pass any valid LiteLLM model string directly
            # e.g. "openai/gpt-4o", "anthropic/claude-3-5-sonnet"
            self.model = model_type

            # ✅ Bug 1 Fix: was referencing undefined `model_alias_lower`, now using `model_lower`
            if "gemini" in model_lower:
                self.fallback_model = "groq/llama-3.3-70b-versatile"
            else:
                self.fallback_model = "gemini/gemini-2.5-flash"

    async def get_streaming_response(self):
        """
        Initiates the asynchronous streaming call to the LLM API.
        
        Attempts to resolve the request using the primary model. If a RateLimitError
        occurs, it retries up to 3 times using an exponential backoff formula (2 ** attempt).
        If all retries fail, or if a connection/API error occurs, it falls back
        to the secondary fallback model.
        
        Returns:
            The async generator stream returned by acompletion().
            
        Raises:
            HTTPException: With descriptive codes and messages if all attempts fail.
        """
        models_to_attempt = [self.model, self.fallback_model]
        last_exception = None
        
        for current_model in models_to_attempt:
            print(f"--- Attempting to call LLM model: '{current_model}' ---")
            
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    api_response_stream = await acompletion(
                        model=current_model,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": self.user_prompt}
                        ],
                        temperature=0.7,
                        # ✅ Bug 2 Fix: was num_retries=1 which caused LiteLLM to retry
                        # internally AND our manual loop also retried — double retrying
                        # is unpredictable and wasteful. Set to 0 so WE control all retries.
                        num_retries=0,
                        stream=True
                    )
                    
                    # Update self.model to whichever model actually succeeded
                    # so that stream_generator logs metrics against the correct provider
                    self.model = current_model
                    print(f"Successfully started stream using model: '{current_model}'")
                    return api_response_stream
                    
                except RateLimitError as rate_err:
                    last_exception = rate_err
                    
                    # On the final attempt, stop waiting and move to fallback model
                    if attempt == max_retries:
                        print(f"Rate limit retry limit ({max_retries}) reached for '{current_model}'. Moving to fallback...")
                        break
                        
                    # Exponential backoff: attempt 1 → 2s, attempt 2 → 4s, attempt 3 → 8s
                    wait_seconds = 2 ** attempt
                    print(f"Rate limit hit on '{current_model}'. "
                          f"Attempt {attempt}/{max_retries}. Retrying in {wait_seconds}s...")
                    
                    # asyncio.sleep() yields control back to the event loop
                    # so other requests are NOT blocked while we wait
                    await asyncio.sleep(wait_seconds)
                    
                except (AuthenticationError, APIConnectionError, APIError, Exception) as other_err:
                    # These errors won't be fixed by retrying the same model
                    # e.g. wrong API key, network down, provider outage
                    # So we skip straight to the fallback model
                    print(f"Non-retryable error on '{current_model}': {other_err}")
                    print("Skipping retries and shifting to fallback model...")
                    last_exception = other_err
                    break
                    
        # If we reach here, both primary and fallback models failed completely
        print("Error: All model attempts and retries have failed.")
        
        # Map the last seen exception type to the appropriate HTTP status code
        # This helps the frontend or API consumer understand what went wrong
        if isinstance(last_exception, RateLimitError):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded on all models. Please try again later."
            )
        elif isinstance(last_exception, AuthenticationError):
            raise HTTPException(
                status_code=401,
                detail="Authentication failed. Please verify API key configuration."
            )
        elif isinstance(last_exception, APIConnectionError):
            raise HTTPException(
                status_code=503,
                detail="Could not connect to any LLM provider. Please check your network."
            )
        else:
            raise HTTPException(
                status_code=502,
                detail=f"LLM execution failed on all models. Error: {str(last_exception)}"
            )

