"""
NICTO Streaming API Server.

FastAPI server for real-time multi-modal generation.
Supports streaming responses for text, image, video, audio, and 3D point clouds.

Endpoints:
  POST /generate/text      - Stream text generation
  POST /generate/image     - Generate images
  POST /generate/video     - Generate video clips
  POST /generate/audio     - Generate audio/music
  POST /generate/3d        - Generate 3D point clouds
  POST /generate/multi     - Generate multiple modalities at once
  GET  /health             - Health check
  GET  /models             - List available models
  WebSocket /ws/generate   - WebSocket for real-time streaming
"""

import asyncio
import json
import time
import uuid
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field


# ==============================================================================
# Request/Response Models
# ==============================================================================

class TextGenRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = True


class ImageGenRequest(BaseModel):
    prompt: Optional[str] = None
    width: int = 512
    height: int = 512
    num_steps: int = 20
    batch_size: int = 1


class VideoGenRequest(BaseModel):
    prompt: Optional[str] = None
    num_frames: int = 16
    width: int = 256
    height: int = 256
    num_steps: int = 10
    fps: int = 8


class AudioGenRequest(BaseModel):
    prompt: Optional[str] = None
    duration: float = 5.0
    sample_rate: int = 22050
    num_steps: int = 10


class PointCloudGenRequest(BaseModel):
    prompt: Optional[str] = None
    num_points: int = 2048
    num_steps: int = 20
    batch_size: int = 1


class MultiGenRequest(BaseModel):
    text: Optional[str] = None
    image: Optional[ImageGenRequest] = None
    video: Optional[VideoGenRequest] = None
    audio: Optional[AudioGenRequest] = None
    point_cloud: Optional[PointCloudGenRequest] = None


class GenerationResponse(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    modality: str
    data: dict
    latency_ms: float
    timestamp: float = Field(default_factory=time.time)


class HealthResponse(BaseModel):
    status: str
    version: str
    modalities: list
    device: str
    uptime_s: float


# ==============================================================================
# Model Manager
# ==============================================================================

class ModelManager:
    """Manages all generation models."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self._loaded = False
        self._start_time = time.time()

    def load_models(self):
        """Lazy load all models."""
        if self._loaded:
            return

        from nicto_ai.nictos.generation.unified_vae import UnifiedVAE
        from nicto_ai.nictos.generation.video_generator import VideoGenerator
        from nicto_ai.nictos.generation.audio_generator import MusicGenerator, SFXGenerator
        from nicto_ai.nictos.generation.point_cloud_generator import PointCloudGenerator
        from nicto_ai.nictos.generation.creativity_engine import CreativityEngine

        print(f"Loading models on {self.device}...")
        t0 = time.time()

        self.models["vae"] = UnifiedVAE(latent_dim=512).to(self.device)
        self.models["video"] = VideoGenerator().to(self.device)
        self.models["music"] = MusicGenerator().to(self.device)
        self.models["sfx"] = SFXGenerator().to(self.device)
        self.models["point_cloud"] = PointCloudGenerator(latent_dim=512).to(self.device)
        self.models["creativity"] = CreativityEngine(latent_dim=512).to(self.device)

        # Set to eval mode
        for model in self.models.values():
            model.eval()

        self._loaded = True
        print(f"Models loaded in {time.time()-t0:.1f}s")

    def get_model(self, name: str):
        if not self._loaded:
            self.load_models()
        return self.models.get(name)

    def uptime(self):
        return time.time() - self._start_time


# Global model manager
manager = ModelManager()


# ==============================================================================
# App Lifecycle
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    manager.load_models()
    yield


app = FastAPI(
    title="NICTO API",
    description="Multi-modal generation API with streaming support",
    version="0.2.0",
    lifespan=lifespan,
)


# ==============================================================================
# Health & Info
# ==============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version="0.2.0",
        modalities=["text", "image", "video", "audio", "point_cloud"],
        device=manager.device,
        uptime_s=manager.uptime(),
    )


@app.get("/models")
async def list_models():
    if not manager._loaded:
        return {"status": "loading"}
    return {
        name: {
            "parameters": sum(p.numel() for p in model.parameters()),
            "device": str(next(model.parameters()).device),
        }
        for name, model in manager.models.items()
    }


# ==============================================================================
# Generation Endpoints
# ==============================================================================

@app.post("/generate/image")
async def generate_image(req: ImageGenRequest):
    """Generate images from text or noise."""
    t0 = time.time()
    vae = manager.get_model("vae")

    with torch.no_grad():
        samples = vae.sample(req.batch_size, modality="image", device=manager.device)

    return GenerationResponse(
        modality="image",
        data={
            "shape": list(samples.shape),
            "min": samples.min().item(),
            "max": samples.max().item(),
        },
        latency_ms=(time.time() - t0) * 1000,
    )


@app.post("/generate/video")
async def generate_video(req: VideoGenRequest):
    """Generate video clips."""
    t0 = time.time()
    video_gen = manager.get_model("video")

    with torch.no_grad():
        video = video_gen.generate(
            num_frames=req.num_frames,
            width=req.width,
            height=req.height,
            num_steps=req.num_steps,
        )

    return GenerationResponse(
        modality="video",
        data={
            "shape": list(video.shape),
            "fps": req.fps,
        },
        latency_ms=(time.time() - t0) * 1000,
    )


@app.post("/generate/audio")
async def generate_audio(req: AudioGenRequest):
    """Generate audio/music."""
    t0 = time.time()
    music_gen = manager.get_model("music")

    with torch.no_grad():
        audio = music_gen.generate(
            duration=req.duration,
            sample_rate=req.sample_rate,
            num_steps=req.num_steps,
        )

    return GenerationResponse(
        modality="audio",
        data={
            "shape": list(audio.shape),
            "sample_rate": req.sample_rate,
            "duration": req.duration,
        },
        latency_ms=(time.time() - t0) * 1000,
    )


@app.post("/generate/3d")
async def generate_point_cloud(req: PointCloudGenRequest):
    """Generate 3D point clouds."""
    t0 = time.time()
    pc_gen = manager.get_model("point_cloud")

    with torch.no_grad():
        points = pc_gen.generate(
            num_points=req.num_points,
            batch_size=req.batch_size,
            num_steps=req.num_steps,
            device=manager.device,
        )

    return GenerationResponse(
        modality="point_cloud",
        data={
            "shape": list(points.shape),
            "num_points": req.num_points,
        },
        latency_ms=(time.time() - t0) * 1000,
    )


@app.post("/generate/multi")
async def generate_multi(req: MultiGenRequest):
    """Generate multiple modalities at once."""
    t0 = time.time()
    results = {}

    if req.image:
        vae = manager.get_model("vae")
        with torch.no_grad():
            results["image"] = vae.sample(1, modality="image", device=manager.device).shape

    if req.video:
        video_gen = manager.get_model("video")
        with torch.no_grad():
            results["video"] = video_gen.generate(num_frames=req.video.num_frames).shape

    if req.audio:
        music_gen = manager.get_model("music")
        with torch.no_grad():
            results["audio"] = music_gen.generate(duration=req.audio.duration).shape

    if req.point_cloud:
        pc_gen = manager.get_model("point_cloud")
        with torch.no_grad():
            results["point_cloud"] = pc_gen.generate(
                num_points=req.point_cloud.num_points,
                device=manager.device,
            ).shape

    return GenerationResponse(
        modality="multi",
        data={"modalities": results},
        latency_ms=(time.time() - t0) * 1000,
    )


# ==============================================================================
# Streaming Text Generation
# ==============================================================================

async def stream_text_chunks(prompt: str, max_tokens: int, temperature: float) -> AsyncGenerator[str, None]:
    """Simulate streaming text generation."""
    # Placeholder: in production, this would use the actual LLM
    response = f"[NICTO] Processing: {prompt}"
    words = response.split()
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        await asyncio.sleep(0.05)  # Simulate generation delay


@app.post("/generate/text")
async def generate_text(req: TextGenRequest):
    """Generate text with optional streaming."""
    if req.stream:
        return StreamingResponse(
            stream_text_chunks(req.prompt, req.max_tokens, req.temperature),
            media_type="text/plain",
        )
    else:
        t0 = time.time()
        full_response = ""
        async for chunk in stream_text_chunks(req.prompt, req.max_tokens, req.temperature):
            full_response += chunk
        return GenerationResponse(
            modality="text",
            data={"text": full_response},
            latency_ms=(time.time() - t0) * 1000,
        )


# ==============================================================================
# WebSocket for Real-Time Streaming
# ==============================================================================

class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active[client_id] = websocket

    def disconnect(self, client_id: str):
        self.active.pop(client_id, None)

    async def send(self, client_id: str, data: dict):
        ws = self.active.get(client_id)
        if ws:
            await ws.send_json(data)


ws_manager = ConnectionManager()


@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """WebSocket endpoint for real-time generation."""
    client_id = str(uuid.uuid4())
    await ws_manager.connect(websocket, client_id)

    try:
        while True:
            data = await websocket.receive_json()
            modality = data.get("modality", "text")

            await ws_manager.send(client_id, {
                "type": "status",
                "modality": modality,
                "message": "Generating...",
            })

            t0 = time.time()

            if modality == "image":
                vae = manager.get_model("vae")
                with torch.no_grad():
                    result = vae.sample(1, modality="image", device=manager.device)
                await ws_manager.send(client_id, {
                    "type": "result",
                    "modality": "image",
                    "shape": list(result.shape),
                    "latency_ms": (time.time() - t0) * 1000,
                })

            elif modality == "point_cloud":
                pc_gen = manager.get_model("point_cloud")
                with torch.no_grad():
                    result = pc_gen.generate(
                        num_points=data.get("num_points", 2048),
                        device=manager.device,
                    )
                await ws_manager.send(client_id, {
                    "type": "result",
                    "modality": "point_cloud",
                    "shape": list(result.shape),
                    "latency_ms": (time.time() - t0) * 1000,
                })

            elif modality == "text":
                async for chunk in stream_text_chunks(
                    data.get("prompt", ""),
                    data.get("max_tokens", 256),
                    data.get("temperature", 0.7),
                ):
                    await ws_manager.send(client_id, {
                        "type": "chunk",
                        "modality": "text",
                        "text": chunk,
                    })
                await ws_manager.send(client_id, {
                    "type": "done",
                    "modality": "text",
                    "latency_ms": (time.time() - t0) * 1000,
                })

    except WebSocketDisconnect:
        ws_manager.disconnect(client_id)


# ==============================================================================
# Run
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
