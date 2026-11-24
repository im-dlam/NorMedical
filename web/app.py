import json
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# Đảm bảo import được package q3AI khi chạy: uvicorn app:app --port 8386 từ thư mục web/
WEB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import build_application

app = FastAPI(title="RaumaMedical q3AI Web", version="1.0.0")
service, api_url = build_application()


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def serve_index():
    return FileResponse(WEB_DIR / "static" / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "knowledge_count": service.knowledge_count,
        "lm_studio_url": api_url,
    }


@app.post("/api/reset")
def reset_chat():
    service.reset_session()
    return {"status": "reset_ok"}


@app.post("/api/chat")
def chat_stream(req: ChatRequest):
    user_msg = req.message.strip()

    def event_generator():
        if not user_msg:
            yield f"event: done\ndata: {json.dumps({'ok': True})}\n\n"
            return

        docs, token_stream = service.stream_turn(user_msg)
        rag_payload = [
            {
                "type": d.doc_type,
                "title": d.title,
                "snippet": d.rag_text[:1000],
            }
            for d in docs
        ]
        yield f"event: rag\ndata: {json.dumps(rag_payload, ensure_ascii=False)}\n\n"

        try:
            for token in token_stream:
                yield f"event: token\ndata: {json.dumps({'text': token}, ensure_ascii=False)}\n\n"
        except ConnectionError as err:
            yield f"event: error\ndata: {json.dumps({'message': str(err)}, ensure_ascii=False)}\n\n"

        yield f"event: done\ndata: {json.dumps({'ok': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8386, reload=False)
