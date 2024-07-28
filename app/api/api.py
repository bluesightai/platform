from fastapi import APIRouter

from app.api.endpoints import auth, embeddings, files, inference, train
from app.utils.logging import LoggingRoute

api_router = APIRouter(route_class=LoggingRoute)


api_router.include_router(train.router, prefix="/train", tags=["Train"])
api_router.include_router(inference.router, prefix="/inference", tags=["Inference"])
api_router.include_router(embeddings.router, prefix="/embeddings", tags=["Embeddings"])
api_router.include_router(files.router, prefix="/files", tags=["Files"])
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"], include_in_schema=False)
