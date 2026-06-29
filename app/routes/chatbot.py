from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.ai.llm_chat import ChatService


router = APIRouter()

@router.post("/chatbot")
async def chatbot(model_name: str, user_prompt: str):
    service = ChatService(model_type = model_name, user_prompt = user_prompt)
    
    # Get the streaming response generator from the ChatService
    stream_generator = await service.get_streaming_response()
    
    # Wrap the generator in FastAPI's StreamingResponse
    return StreamingResponse(
        stream_generator,      # The async generator that yields text chunks
        media_type="text/event-stream",  #SSE(Server Sent Events) format
        status_code=200,
    )

