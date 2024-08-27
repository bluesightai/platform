"""
Modified original inference script:
https://github.com/CASIA-IVA-Lab/FastSAM/blob/main/Inference.py
"""

from typing import Literal

import fire
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import FastSAM

from app.core.fastsam_prompt import FastSAMPrompt


def convert_box_xywh_to_xyxy(box):
    if len(box) == 4:
        return [box[0], box[1], box[0] + box[2], box[1] + box[3]]
    else:
        return [convert_box_xywh_to_xyxy(b) for b in box]


def main(
    model_filename: Literal["FastSAM-x.pt", "FastSAM-s.pt"] = "FastSAM-x.pt",
    img_path: str = "./data/samples/pool.png",
    imgsz: int = 256,
    iou: float = 0.9,
    text_prompt: str = "a blue swimming pool",
    text_prompt_top_k: int = 5,
    conf: float = 0.4,
    output: str = "./data/fastsam_output/",
    point_prompt: list[tuple[int, int]] = [(0, 0)],
    point_label: list[int] = [0],
    box_prompt: list[tuple[int, int, int, int]] = [(0, 0, 0, 0)],
    better_quality: bool = True,
    device: str | None = None,
    retina: bool = True,
    withContours: bool = True,
):
    """
    Run FastSAM (Fast Segment Anything Model) on an image.

    Args:
        model_filename: Model filename to download from Hugging Face Hub.
        img_path: Path to the input image file.
        imgsz: Image size for processing.
        iou: IoU threshold for filtering annotations.
        text_prompt: Text prompt for object detection (e.g., "a dog").
        conf: Object confidence threshold.
        output: Path to save the output image.
        point_prompt: List of point prompts as tuples, e.g., [(x1, y1), (x2, y2)].
        point_label: List of labels for point prompts. 0 for background, 1 for foreground.
        box_prompt: List of box prompts as tuples (x, y, w, h), e.g., [(x1, y1, w1, h1), (x2, y2, w2, h2)].
        better_quality: Use morphologyEx for better quality masks.
        device: Device to run the model on (cuda, mps, or cpu).
        retina: Draw high-resolution segmentation masks.
        withContours: Draw the edges of the masks.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model_file = hf_hub_download(repo_id="furiousteabag/FastSAM", filename=model_filename)
    model = FastSAM(model_file)
    box_prompt = convert_box_xywh_to_xyxy(box_prompt)

    input_image = Image.open(img_path).convert("RGB")
    everything_results = model(input_image, device=device, retina_masks=retina, imgsz=imgsz, conf=conf, iou=iou)

    prompt_process = FastSAMPrompt(input_image, everything_results, device=device)

    bboxes = None
    points = None
    if box_prompt[0][2] != 0 and box_prompt[0][3] != 0:
        ann = prompt_process.box_prompt(bboxes=box_prompt)
        bboxes = box_prompt
    elif text_prompt:
        import time

        start_time = time.time()
        ann = prompt_process.text_prompt(text=text_prompt, top_k=text_prompt_top_k)
        end_time = time.time()
        print(f"Time taken to generate annotations for text prompt: {end_time - start_time:.4f} seconds")
    elif point_prompt[0] != (0, 0):
        ann = prompt_process.point_prompt(points=point_prompt, pointlabel=point_label)
        points = point_prompt
    else:
        ann = prompt_process.everything_prompt()

    prompt_process.plot(
        annotations=ann,
        output_path=f"{output}{img_path.split('/')[-1]}",
        bboxes=bboxes,
        points=points,
        point_label=point_label,
        withContours=withContours,
        better_quality=better_quality,
    )


if __name__ == "__main__":
    fire.Fire(main)
