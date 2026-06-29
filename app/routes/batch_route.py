from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List
from app.ai.llm_batch import BatchProcessor

router = APIRouter()


class BatchRunRequest(BaseModel):
    prompts: List[str] = Field(..., min_items=1)
    model_name: str = "fast"
    use_chain: bool = False
    max_concurrency: int = 3


@router.post("/batch/run")
async def start_batch(request: BatchRunRequest, background_tasks: BackgroundTasks):
    # Initialize the job structure on disk and obtain job_id
    job_id = await BatchProcessor.start_batch_job(
        prompts=request.prompts,
        model_key=request.model_name,
        use_chain=request.use_chain,
        max_concurrency=request.max_concurrency
    )

    # Enqueue execution in FastAPI background tasks
    background_tasks.add_task(
        BatchProcessor.run_batch_execution,
        job_id=job_id,
        prompts=request.prompts,
        model_key=request.model_name,
        use_chain=request.use_chain,
        max_concurrency=request.max_concurrency
    )

    return {
        "success": True,
        "job_id": job_id,
        "message": "Batch job successfully queued in background.",
        "status_url": f"/batch/status/{job_id}"
    }


@router.get("/batch/status/{job_id}")
async def get_batch_status(job_id: str):
    job_data = BatchProcessor.load_job(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="Batch job not found.")
    return job_data
