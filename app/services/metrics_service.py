import json
from datetime import datetime, timezone
from litellm import completion_cost

class MetricsTracker:
    """
    Service class responsible for tracking, printing, and parsing metrics 
    from LiteLLM completion responses.
    """
    
    @staticmethod
    def log_metrics(response, model_name: str, **kwargs) -> dict:
        """
        Extracts, prints, and returns structured metrics from the LiteLLM response.
        
        Args:
            response: The raw completion response object from LiteLLM.
            model_name: The identifier of the model that was called.
            
        Returns:
            A dictionary containing timestamp, model, token usage, estimated cost, and rate limits.
        """
        try:
            #response.usage
            usage = getattr(response, "usage", None)
            headers = getattr(response, "_response_headers", {}) or {}
            
            # Safely calculate cost using litellm
            cost = 0.0
            try:
                cost = completion_cost(completion_response=response, model=model_name) or 0.0
                
            except Exception as cost_err:
                # If litellm fails to compute cost, print a warning but keep default 0.0
                print(f"Warning: Could not compute completion cost for model '{model_name}': {cost_err}")
                
            metrics = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": model_name,
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                },
                "cost": cost,
                "rate_limits": {
                    "remaining_requests": headers.get("x-ratelimit-remaining-requests") 
                                       or headers.get("x-goog-ratelimit-remaining-requests")
                                       or headers.get("ratelimit-remaining")
                                       or headers.get("x-ratelimit-remaining"),
                    "remaining_tokens": headers.get("x-ratelimit-remaining-tokens") 
                                     or headers.get("x-goog-ratelimit-remaining-tokens")
                                     or headers.get("ratelimit-remaining-tokens")
                                     or headers.get("x-ratelimit-remaining-tokens"),
                }
            }
            
            # Print logs for visibility in developer console
            print("--- LLM API Call Metrics ---")
            print(json.dumps(metrics, indent=2))
            
            return metrics
            
        except Exception as e:
            print(f"Metrics logging failed: {e}")
            return {}

