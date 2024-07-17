import pickle
import secrets
import string
from datetime import datetime
from typing import List, Tuple

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from gotrue import UserResponse

from app.config import config, supabase
from app.schemas.auth import LoginData, Token, UserData
from app.schemas.clay import (
    ClassificationLabels,
    Embeddings,
    Images,
    InferClassificationData,
    Points,
    TrainClassificationData,
    TrainResults,
)
from app.schemas.common import TextResponse
from app.utils.auth import get_current_user
from app.utils.logging import LoggingRoute
from clay.model import get_embeddings_img
from clay.train import predict_classification, train_classification

api_router = APIRouter(route_class=LoggingRoute)


@api_router.post("/train/classification", tags=["Train"])
async def train_classification_model(data: TrainClassificationData) -> TrainResults:
    """Train classification model on your data."""
    embeddings = await get_embeddings_with_images(data)
    model, details = train_classification(embeddings=np.array(embeddings.embeddings), labels=np.array(data.labels))
    model_bytes = pickle.dumps(model)
    model_name = f"classification_{len(embeddings.embeddings)}_{''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(5))}"
    response = supabase.storage.from_(config.SUPABASE_MODEL_BUCKET).upload(path=model_name + ".pkl", file=model_bytes)
    return TrainResults(model_id=model_name, train_details=details)


@api_router.post("/inference/classification", tags=["Train"])
async def infer_classification_model(data: InferClassificationData) -> ClassificationLabels:
    """Run inference of previously trained classification model on your data."""
    embeddings = await get_embeddings_with_images(data)
    model = pickle.loads(supabase.storage.from_(config.SUPABASE_MODEL_BUCKET).download(path=data.model_id + ".pkl"))
    labels = predict_classification(clf=model, embeddings=np.array(embeddings.embeddings))
    return ClassificationLabels(labels=labels.tolist())


@api_router.post("/embeddings/img", tags=["Embeddings"])
async def get_embeddings_with_images(images: Images) -> Embeddings:
    """Get embeddings for a list of images."""
    pixels: List[List[List[List[float]]]] = []
    points: List[Tuple[float, float]] = []
    datetimes: List[datetime] = []
    # Check consistency of platform, gsd, and bands
    first_image = images.images[0]
    platform, gsd, bands = first_image.platform, first_image.gsd, first_image.bands
    pixel_shape = None
    for image in images.images:
        if image.platform != platform:
            raise ValueError("Inconsistent platform across images")
        if image.gsd != gsd:
            raise ValueError("Inconsistent gsd across images")
        if image.bands != bands:
            raise ValueError("Inconsistent bands across images")

        if pixel_shape is None:
            pixel_shape = len(image.pixels), len(image.pixels[0]), len(image.pixels[0][0])
        elif (len(image.pixels), len(image.pixels[0]), len(image.pixels[0][0])) != pixel_shape:
            raise ValueError("Inconsistent pixel shapes across images")

        pixels.append(image.pixels)
        points.append(image.point)
        datetimes.append(datetime.fromtimestamp(image.timestamp))
    embeddings = get_embeddings_img(
        platform=images.images[0].platform,
        gsd=images.images[0].gsd,
        bands=images.images[0].bands,
        pixels=pixels,
        points=points,
        datetimes=datetimes,
    ).tolist()
    return Embeddings(embeddings=embeddings)


# @api_router.post("/embeddings/loc", tags=["Embeddings"])
# async def get_embeddings_with_coordinates(points: Points) -> Embeddings:
#     """Get embeddings for a list of points."""
#     return Embeddings(
#         embeddings=[
#             get_embedding(lat=lat, lon=lon, size=points.size, gsd=0.6, start="2022-01-01")[0].squeeze().tolist()
#             for lat, lon in points.points
#         ]
#     )


@api_router.post("/auth/register", tags=["Authentication"], include_in_schema=False)
async def register(user_data: UserData) -> TextResponse:
    """Login endpoint

    ## This is login endpoint
    """
    try:
        response = supabase.auth.sign_up(
            {
                "email": user_data.email,
                "password": user_data.password,
                "options": {"data": {"first_name": user_data.first_name, "last_name": user_data.last_name}},
            }
        )
        return TextResponse(message="Check your email for verification!")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@api_router.get("/auth/token", response_model=Token, tags=["Authentication"], include_in_schema=False)
async def get_token(login_data: LoginData) -> Token:
    try:
        response = supabase.auth.sign_in_with_password({"email": login_data.email, "password": login_data.password})
        if not response.session:
            raise HTTPException(status_code=400, detail="User not found")
        access_token = response.session.access_token
        return Token(access_token=access_token, token_type="bearer")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@api_router.get("/auth/me", tags=["Authentication"], include_in_schema=False)
async def get_my_data(current_user: UserResponse = Depends(get_current_user)):
    return current_user


# @api_router.post("/inference/classification", tags=["inference"])
# async def inference_classification(model_id: str, images: List[ImageData]) -> List[int]:
#     """Embeddings endpoint"""
#     return [2, 1]
