from fastapi import APIRouter

from app.api.endpoints import auth, embeddings, files, inference, models, training_jobs
from app.utils.logging import LoggingRoute

api_router = APIRouter(route_class=LoggingRoute)


@api_router.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to Bluesight API! Documentation available at https://docs.bluesight.ai/api-reference."}


api_router.include_router(training_jobs.router, prefix="/training/jobs", tags=["Training Jobs"])
api_router.include_router(inference.router, prefix="/inference", tags=["Inference"])
api_router.include_router(embeddings.router, prefix="/embeddings", tags=["Embeddings"])
api_router.include_router(files.router, prefix="/files", tags=["Files"])
api_router.include_router(models.router, prefix="/models", tags=["Models"])
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"], include_in_schema=False)
