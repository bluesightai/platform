from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.api import api_router
from app.core.events import lifespan


def create_app() -> FastAPI:
    app = FastAPI(
        lifespan=lifespan,
        servers=[{"url": "https://api.bluesight.ai/"}],
        description="The Bluesight API enables you to create embeddings using the Clay foundation model and run training jobs on top of Clay.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    return app


app = create_app()
