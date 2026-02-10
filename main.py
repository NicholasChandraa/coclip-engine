from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.utils.logging import logger
from app.tools.transcriber import transcriber
from app.core.config import settings
from app.core.database import init_db
from app.api.routes import transcribe
import uvicorn


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Engine is starting up...")

    # Load Whisper model saat startup (optional - bisa di-skip kalau mau lazy loading)
    try:
        transcriber.load_model()
    except Exception as e:
        logger.warning(f"⚠️ Failed to preload model: {e}")
        logger.info("📌 Model will be loaded on first request")

    # Initialize database tables
    try:
        await init_db()
    except Exception as e:
        logger.warning(f"⚠️ Database init failed: {e}")
        logger.info("📌 DB features will be unavailable")

    yield
    logger.info("🛑 Engine is shutting down...")


app = FastAPI(title="CoClip AI Engine", lifespan=lifespan)

# Include routers
app.include_router(
    transcribe.router, prefix=settings.API_V1_STR, tags=["Transcription"]
)


@app.get("/")
async def health_check():
    logger.info("Health check endpoint called")
    return {
        "status": "ok",
        "service": "coclip-engine",
        "model_loaded": transcriber._model is not None,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
