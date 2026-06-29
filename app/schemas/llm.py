from pydantic import BaseModel, Field
from typing import Optional, Any, Dict

class ChatRequest(BaseModel):
    """
    Validation schema for chatbot requests.
    """
    user_prompt: str = Field()
    model_type: str = Field(
        "fast", 
        example="fast", 
        description="The alias or direct identifier of the model to use: 'fast' (Gemini) or 'pro' (Groq)."
    )

class TokenUsage(BaseModel):
    """
    Metadata schema for token usage metrics.
    """
    prompt_tokens: int = Field(0, description="Tokens used for the prompt.")
    completion_tokens: int = Field(0, description="Tokens used for generating the completion.")
    total_tokens: int = Field(0, description="Total tokens used (prompt + completion).")


class RateLimits(BaseModel):
    """
    Metadata schema for tracking remaining rate limits.
    """
    remaining_requests: Optional[Any] = Field(None, description="Number of remaining requests allowed in the current window.")
    remaining_tokens: Optional[Any] = Field(None, description="Number of remaining tokens allowed in the current window.")


class ChatMetrics(BaseModel):
    """
    Schema for full performance and billing metrics.
    """
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of the LLM call.")
    model: str = Field(..., description="The model name that processed the request.")
    usage: TokenUsage = Field(..., description="Breakdown of token counts.")
    cost: float = Field(0.0, description="Calculated API cost of the completion in USD.")
    rate_limits: RateLimits = Field(..., description="Available rate limits remaining on the server headers.")


class ChatResponse(BaseModel):
    """
    Standard schema for chatbot responses returned to the client.
    """
    success: bool = Field(True, description="Indicates whether the request was completed successfully.")
    response: str = Field(..., description="The generated text content from the language model.")
    metrics: Optional[ChatMetrics] = Field(None, description="Performance and usage metrics from the API call.")
