import os
import json
import uuid
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from app.ai.llm_chain import ChainService, call_llm_non_streaming

logger = logging.getLogger(__name__)

# Base directory for job status storage
BATCH_LOG_DIR = Path(__file__).resolve().parent.parent / "logs" / "batches"
BATCH_LOG_DIR.mkdir(parents=True, exist_ok=True)


class BatchProcessor:
    @staticmethod
    def get_job_path(job_id: str) -> Path:
        return BATCH_LOG_DIR / f"job_{job_id}.json"

    @classmethod
    def load_job(cls, job_id: str) -> dict:
        path = cls.get_job_path(job_id)
        if not path.exists():
            return None
        with open(path, "r") as f:
            return json.load(f)

    @classmethod
    def save_job(cls, job_id: str, job_data: dict):
        path = cls.get_job_path(job_id)
        with open(path, "w") as f:
            json.dump(job_data, f, indent=4)

    @classmethod
    async def process_single_item(
        cls,
        item_index: int,
        prompt: str,
        model_key: str,
        use_chain: bool,
        semaphore: asyncio.Semaphore,
        job_id: str
    ):
        """
        Processes a single batch item (either via single LLM call or chain).
        """
        async with semaphore:
            logger.info(f"Job {job_id}: Processing item {item_index}...")
            result_item = {
                "index": item_index,
                "input": prompt,
                "completed_at": None,
                "status": "processing"
            }

            try:
                if use_chain:
                    # Run full extraction/enrichment/formatting chain
                    chain_result = await ChainService.run_concept_chain(prompt, model_key)
                    if chain_result["success"]:
                        result_item["output"] = chain_result["final_report"]
                        result_item["status"] = "success"
                        result_item["execution_history"] = chain_result["execution_history"]
                    else:
                        result_item["error"] = chain_result["error"]
                        result_item["status"] = "failed"
                else:
                    # Run a single direct LLM call
                    messages = [{"role": "user", "content": prompt}]
                    output_text = await call_llm_non_streaming(model_key, messages)
                    result_item["output"] = output_text
                    result_item["status"] = "success"
            except Exception as e:
                logger.error(f"Job {job_id}: Item {item_index} failed: {e}")
                result_item["status"] = "failed"
                result_item["error"] = str(e)

            result_item["completed_at"] = datetime.now(timezone.utc).isoformat()

            # Update job state on disk
            job_data = cls.load_job(job_id)
            if job_data:
                job_data["results"][item_index] = result_item
                job_data["progress"]["completed_items"] += 1

                # Check if all completed
                if job_data["progress"]["completed_items"] == job_data["progress"]["total_items"]:
                    job_data["status"] = "completed"
                    job_data["completed_at"] = datetime.now(timezone.utc).isoformat()

                cls.save_job(job_id, job_data)

    @classmethod
    async def start_batch_job(
        cls,
        prompts: list,
        model_key: str = "fast",
        use_chain: bool = False,
        max_concurrency: int = 3
    ) -> str:
        """
        Initializes a batch job and returns the job_id immediately.
        The caller should run the actual execution as a background task.
        """
        job_id = str(uuid.uuid4())
        total_items = len(prompts)

        job_data = {
            "job_id": job_id,
            "status": "pending",
            "model_key": model_key,
            "use_chain": use_chain,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "progress": {
                "total_items": total_items,
                "completed_items": 0
            },
            "results": [None] * total_items
        }

        cls.save_job(job_id, job_data)
        return job_id

    @classmethod
    async def run_batch_execution(
        cls,
        job_id: str,
        prompts: list,
        model_key: str,
        use_chain: bool,
        max_concurrency: int = 3
    ):
        """
        Executes the batch processing of prompts in parallel using a Semaphore.
        """
        job_data = cls.load_job(job_id)
        if not job_data:
            logger.error(f"Cannot run batch execution. Job {job_id} not found.")
            return

        job_data["status"] = "processing"
        cls.save_job(job_id, job_data)

        semaphore = asyncio.Semaphore(max_concurrency)
        tasks = []
        for index, prompt in enumerate(prompts):
            task = asyncio.create_task(
                cls.process_single_item(
                    item_index=index,
                    prompt=prompt,
                    model_key=model_key,
                    use_chain=use_chain,
                    semaphore=semaphore,
                    job_id=job_id
                )
            )
            tasks.append(task)

        try:
            await asyncio.gather(*tasks)
        except Exception as gather_err:
            logger.error(f"Error gathering batch job {job_id}: {gather_err}")
            job_data = cls.load_job(job_id)
            if job_data and job_data["status"] != "completed":
                job_data["status"] = "failed"
                job_data["completed_at"] = datetime.now(timezone.utc).isoformat()
                cls.save_job(job_id, job_data)

