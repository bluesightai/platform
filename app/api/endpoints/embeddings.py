import base64
import io
from datetime import datetime
from typing import List, Tuple

import numpy as np
from fastapi import APIRouter
from numpy.typing import NDArray

from app.api.deps import SessionDep
from app.schemas.clay import Embeddings, EmbeddingsRequestBase, Points
from app.utils.logging import LoggingRoute
from clay.clip import get_embeddings_from_images, get_embeddings_from_text
from clay.model import get_embedding, get_embeddings_img
from clay.utils import get_catalog_items, get_stack, stack_to_datacube

router = APIRouter(route_class=LoggingRoute)


@router.post("/img")
async def get_embeddings_with_images(images: EmbeddingsRequestBase, session: SessionDep) -> Embeddings:
    """Get embeddings for a list of images."""
    pixels: List[NDArray[np.uint8]] = []
    points: List[Tuple[float, float] | None] = []
    datetimes: List[datetime | None] = []
    # Check consistency of platform, gsd, and bands
    first_image = images.images[0]
    platform, gsd, bands = first_image.platform, first_image.gsd, first_image.bands
    pixel_shape = None
    for image in images.images:
        # if image.platform != platform:
        #     raise ValueError("Inconsistent platform across images")
        # if image.gsd != gsd:
        #     raise ValueError("Inconsistent gsd across images")
        # if image.bands != bands:
        #     raise ValueError("Inconsistent bands across images")

        # if pixel_shape is None:
        #     pixel_shape = len(image.pixels), len(image.pixels[0]), len(image.pixels[0][0])
        # elif (len(image.pixels), len(image.pixels[0]), len(image.pixels[0][0])) != pixel_shape:
        #     raise ValueError("Inconsistent pixel shapes across images")

        img_array = np.load(io.BytesIO(base64.b64decode(image.pixels)))
        if img_array.shape[0] != len(bands):
            raise ValueError(f"Expected {len(bands)} bands, got {img_array.shape[0]}")
        pixels.append(img_array)
        points.append(image.point)
        datetimes.append(datetime.fromtimestamp(image.timestamp) if image.timestamp else None)

    if images.model == "clip":
        embeddings = get_embeddings_from_images(images=pixels)
    elif images.model == "clay":
        embeddings = get_embeddings_img(
            gsd=images.images[0].gsd,
            bands=images.images[0].bands,
            pixels=pixels,
            platform=images.images[0].platform,
            wavelengths=images.images[0].wavelengths,
            points=points,
            datetimes=datetimes,
        ).tolist()
    else:
        raise ValueError("Invalid model")

    return Embeddings(embeddings=embeddings)


@router.post("/loc")
async def get_embeddings_with_coordinates(points: Points, session: SessionDep) -> Embeddings:
    """Get embeddings for a list of points."""
    items = [get_catalog_items(lat=lat, lon=lon, start="2022-01-01") for lat, lon in points.points]
    stacks = [
        get_stack(lat=lat, lon=lon, items=item, size=points.size, gsd=0.6)
        for item, (lat, lon) in zip(items, points.points)
    ]
    datacubes = [stack_to_datacube(lat=lat, lon=lon, stack=stack) for stack, (lat, lon) in zip(stacks, points.points)]
    return Embeddings(embeddings=[get_embedding(datacube=datacube).tolist() for datacube in datacubes])


@router.post("/text")
async def get_embeddings_with_text(texts: list[str], session: SessionDep) -> Embeddings:
    """Get embeddings for a list of texts."""
    embeddings = get_embeddings_from_text(texts=texts)
    return Embeddings(embeddings=embeddings)
