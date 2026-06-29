import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import litellm

# Enable returning API response headers for rate limits
litellm.return_response_headers = True
logger = logging.getLogger(__name__)

class MetricsTracker:
    def __init__(self, log_dir: str = None):
        if log_dir is None:
            # Resolve the folder where this script lives (e.g., app/ai/)
            current_folder = Path(__file__).resolve().parent
            # Go up one level to 'app/', and locate the 'logs/' folder
            self.log_dir = current_folder.parent / "logs"
        else:
            self.log_dir = Path(log_dir)
        
        # Create the logs directory if it doesn't already exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Path to the metrics JSON file
        self.log_file_path = self.log_dir / "metrics_log.json"

    def log_metrics(self, model_name: str, response, **kwargs) -> dict:
        """
        Extracts token usage and rate limits from the LiteLLM response and writes them to a JSON file.
        """
        try:
            usage = getattr(response, "usage", None)
            headers = getattr(response, "_response_headers", None) or {}

            rate_limits = {
                "remaining_requests": headers.get("x-ratelimit-remaining-requests") or headers.get("x-goog-ratelimit-remaining-requests"),
                "remaining_tokens": headers.get("x-ratelimit-remaining-tokens") or headers.get("x-goog-ratelimit-remaining-tokens"),
                "limit_requests": headers.get("x-ratelimit-limit-requests") or headers.get("x-goog-ratelimit-limit-requests"),
                "limit_tokens": headers.get("x-ratelimit-limit-tokens") or headers.get("x-goog-ratelimit-limit-tokens"),
                "reset_requests": headers.get("x-ratelimit-reset-requests") or headers.get("x-goog-ratelimit-reset-requests"),
                "reset_tokens": headers.get("x-ratelimit-reset-tokens") or headers.get("x-goog-ratelimit-reset-tokens"),
            }

            usage_metrics = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0)
            }

            metrics = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": model_name,
                "usage": usage_metrics,
                "rate_limits": rate_limits
            }

            # File writing with error containment
            log_data = []
            if os.path.exists(self.log_file_path):
                try:
                    with open(self.log_file_path, "r") as f:
                        content = f.read().strip()
                        if content:
                            log_data = json.loads(content)
                except (json.JSONDecodeError, IOError) as e:
                    logger.error(f"Error reading existing metrics log: {e}")
            
            log_data.append(metrics)

            with open(self.log_file_path, "w") as f:
                json.dump(log_data, f, indent=4)

        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
            return {"status": "error", "message": str(e)}

        return {
            "status": "success", 
            "log_path": self.log_file_path,
            "metrics": metrics
        }

metrics = MetricsTracker()
