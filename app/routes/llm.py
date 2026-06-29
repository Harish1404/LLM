from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas.llm import ChatRequest
from app.services.llm_service import ChatService, stream_generator

router = APIRouter()

@router.post("/chatbot")
async def chatbot(request: ChatRequest):

    service = ChatService(model_type=request.model_type,user_prompt=request.user_prompt)
    api_response_stream = await service.get_streaming_response()

    return StreamingResponse(stream_generator(api_response_stream, service.model), media_type="text/event-stream")



