from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.api import api_router


def create_app() -> FastAPI:
    app = FastAPI(servers=[{"url": "https://api.bluesight.ai/"}])

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include the routers
    app.include_router(api_router)

    return app


app = create_app()
