import json
from typing import Literal

from app.schemas.clay import ClassificationTrainingDataSample, SegmentationTrainingDataSample


def validate_jsonl(file_path: str, task: Literal["classification", "segmentation"]):
    with open(file_path, "r") as file:
        for line_num, line in enumerate(file, 1):
            try:
                data = json.loads(line)
                if task == "classification":
                    ClassificationTrainingDataSample(**data)
                elif task == "segmentation":
                    SegmentationTrainingDataSample(**data)
                else:
                    raise ValueError(f"Invalid task: {task}")
            except Exception as e:
                raise TypeError(f"File format validation failed on line {line_num}: {e.__class__.__name__}: {e}")
