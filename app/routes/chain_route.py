from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.ai.llm_chain import ChainService

router = APIRouter()


class ChainRequest(BaseModel):
    raw_text: str
    model_name: str = "fast"


@router.post("/chain")
async def run_chain(request: ChainRequest):
    if not request.raw_text.strip():
        raise HTTPException(status_code=400, detail="raw_text cannot be empty.")

    result = await ChainService.run_concept_chain(
        raw_text=request.raw_text,
        model_key=request.model_name
    )

    if not result["success"]:
        raise HTTPException(status_code=502, detail=result)

    return result
