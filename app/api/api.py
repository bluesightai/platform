from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from gotrue import UserResponse
from loguru import logger

from app.config import supabase
from app.schemas.auth import LoginData, Token, UserData
from app.schemas.clay import Embeddings, Images, Points
from app.schemas.common import TextResponse
from app.utils.auth import get_current_user
from app.utils.logging import LoggingRoute
from clay.model import get_embedding, get_embedding_img

api_router = APIRouter(route_class=LoggingRoute)


@api_router.post("/embeddings/loc", tags=["Embeddings"])
async def get_embeddings_with_coordinates(points: Points) -> Embeddings:
    """Get embeddings for a list of points."""
    return Embeddings(
        embeddings=[
            get_embedding(lat=lat, lon=lon, size=128, gsd=0.6, start="2022-01-01")[0].squeeze().tolist()
            for lat, lon in points.points
        ]
    )


@api_router.post("/embeddings/img", tags=["Embeddings"])
async def get_embeddings_with_images(images: Images) -> Embeddings:
    """Get embeddings for a list of images."""
    return Embeddings(
        embeddings=[
            get_embedding_img(
                platform=image.platform,
                pixels=image.pixels,
                bands=image.bands,
                point=image.point,
                timestamp=datetime.fromtimestamp(image.timestamp),
                gsd=image.gsd,
            )
            .squeeze()
            .tolist()
            for image in images.images
        ]
    )


@api_router.post("/auth/register", tags=["Authentication"])
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


@api_router.get("/auth/token", response_model=Token, tags=["Authentication"])
async def get_token(login_data: LoginData) -> Token:
    try:
        response = supabase.auth.sign_in_with_password({"email": login_data.email, "password": login_data.password})
        if not response.session:
            raise HTTPException(status_code=400, detail="User not found")
        access_token = response.session.access_token
        return Token(access_token=access_token, token_type="bearer")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@api_router.get("/auth/me", tags=["Authentication"])
async def get_my_data(current_user: UserResponse = Depends(get_current_user)):
    return current_user


# @api_router.post("/train/classification", tags=["train"])
# async def train_classification(images: List[ImageData], labels: List[int]) -> str:
#     """Embeddings endpoint"""
#     return "slkdfj12l3jlj"


# @api_router.post("/inference/classification", tags=["inference"])
# async def inference_classification(model_id: str, images: List[ImageData]) -> List[int]:
#     """Embeddings endpoint"""
#     return [2, 1]
