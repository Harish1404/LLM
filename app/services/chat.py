from litellm import acompletion , stream_chunk_builder
from app.prompts.system_prompt import SYSTEM_PROMPT
from litellm.exceptions import RateLimitError, APIError
import random, asyncio, logging 
from app.core.config import settings


logger = logging.getlogger(__name__)


class ChatService:
    
    def __init__(self, model_type: str, user_prompt: str):

        self.model_type = model_type
        self.user_prompt = user_prompt

        self.gemeini_api = settings.GEMINI_API_KEY
        self.groq_api = settings.GROQ_API_KEY

        self.fallback_chain = [
            ("gemini/gemini-2.5-flash", self.gemeini_api ),
            ("groq/llama-3.1-8b-instant", self.groq_api)
        ]

        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT  },
            {"role": "user", "content": self.user_prompt}
        ]

    async def stream_retry(model: str, api_key, messages: list, max_retries: int = 3 , **kwargs,):
        
        for attempts in range(1, max_retries + 1):

            try:
                response = acompletion(
                    self,
                    model = model,
                    api_key = api_key,
                    messages = messages,
                    temperature = kwargs["temperature"],
                    max_tokens = kwargs["max_tokens"],
                    stream = True
                )

                

                collected_chunk_list = []

                try:

                    async for chunk in response:
                        collected_chunk_list.append(chunk)

                        choices = getattr(chunk, "choices", [])

                        if choices:
                            delta_content = getattr(choices[0].delta, "content", "")

                            if delta_content:
                                yield delta_content

                    return
                except Exception as stream_err:
                    logger.error("stremaing error occures")

                finally:

                    if collected_chunk_list:
                         collected_repsonse =  stream_chunk_builder(collected_chunk_list)
            
            except RateLimitError as rate_error:

                if attempts == max_retries - 2:

                    headers = getattr(rate_error, "header", None)

                    retry_after_header = headers.get("retry-after")

                    if retry_after_header:
                        wait = 2**attempts 

                await asyncio.sleep(wait)

            except APIError as api_err:
          
                logger.error(f"[{model}] non retryable api error..({api_err})")
                raise api_err
    

    async def chat(self):

        for model_type, api_key in self.fallback_chain:
            try:
                async for chunk in self.stream_retry(
                    model = model_type,
                    api_key = api_key,
                    messages = self.messages,
                    temperature = 0.7,
                    max_tokens = 100

                ):
                    yield chunk

                return

            except Exception as e:
                logger.error(f"{model_type} this is not working...")

                


