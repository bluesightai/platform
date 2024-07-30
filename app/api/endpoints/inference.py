import pickle

import numpy as np
from fastapi import APIRouter

from app.api.endpoints.embeddings import get_embeddings_with_images
from app.config import config, supabase
from app.schemas.clay import ClassificationLabels, Images, InferenceData, SegmentationLabels
from app.utils.logging import LoggingRoute
from clay.train import predict_classification

router = APIRouter(route_class=LoggingRoute)


@router.post("")
async def run_trained_model_inference(inference_data: InferenceData) -> ClassificationLabels | SegmentationLabels:
    """Run inference of previously trained classification model on your data."""

    model_path = config.FILES_CACHE_DIR / inference_data.model
    if not model_path.exists():
        with open(model_path, "wb") as f:
            f.write(supabase.storage.from_(config.SUPABASE_MODEL_BUCKET).download(path=inference_data.model))

    if "classification" in inference_data.model:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        embeddings = await get_embeddings_with_images(Images(images=inference_data.images))
        labels = predict_classification(clf=model, embeddings=np.array(embeddings.embeddings))
        return ClassificationLabels(labels=labels.tolist())
    elif "segmentation" in inference_data.model:
        return SegmentationLabels(labels=[[]])
    else:
        raise ValueError(f"Unknown model type: {inference_data.model}. Supported models: classification, segmentation")


@router.post("/segmentation")
async def infer_segmentation_model(data: InferenceData) -> SegmentationLabels:
    """Run inference of previously trained segmentation model on your data."""

    checkpoint_download_path = f".cache/{data.model}"
    os.makedirs(os.path.dirname(checkpoint_download_path), exist_ok=True)
    with open(checkpoint_download_path, "wb+") as f:
        res = supabase.storage.from_(config.SUPABASE_MODEL_BUCKET).download(data.model)
        f.write(res)

    pixels: List[List[List[List[float]]]] = []
    points: List[Tuple[float, float] | None] = []
    datetimes: List[datetime | None] = []
    # Check consistency of platform, gsd, and bands
    first_image = data.images[0]
    platform, gsd, bands = first_image.platform, first_image.gsd, first_image.bands
    pixel_shape = None
    for image in data.images:
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
        datetimes.append(datetime.fromtimestamp(image.timestamp) if image.timestamp else None)

    # model = SegmentorTraining.load_from_checkpoint("./checkpoints/segment/last.ckpt")
    model = SegmentorTraining.load_from_checkpoint(checkpoint_download_path)
    data_module = SegmentationDataModule(
        gsd=data.images[0].gsd,
        bands=data.images[0].bands,
        pixels=pixels,
        platform=data.images[0].platform,
        wavelengths=data.images[0].wavelengths,
        points=points,
        datetimes=datetimes,
        batch_size=40,
        num_workers=0,
        train_test_ratio=0,
        labels=None,
    )
    data_module.setup()

    all_preds = []
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for batch in data_module.val_dataloader():
            outputs = model(batch)
            outputs = F.interpolate(
                outputs,
                size=(batch["pixels"].shape[-2], batch["pixels"].shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_preds.append(preds)

    # Combine all predictions
    all_preds = np.concatenate(all_preds, axis=0)

    logger.warning(all_preds.shape)

    return SegmentationLabels(labels=[all_preds.tolist()[: len(pixels)]])
