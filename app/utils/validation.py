import json
from typing import Literal

import h5py
import numpy as np

from app.schemas.clay import ClassificationTrainingDataSample, SegmentationTrainingDataSample


def validate_hdf5(file_path: str, task: Literal["classification", "segmentation"]):
    with h5py.File(file_path, "r") as f:

        if "data" not in f:
            raise ValueError("Dataset 'data' not found in the file.")

        dataset: h5py.Dataset = f["data"]

        first_item = dataset[0]

        expected_dtypes = [
            ("bands", "S10"),
            ("gsd", "float32"),
            ("pixels", "float32"),
            ("platform", "S20"),
            ("point", "float32"),
            ("timestamp", "int64"),
            ("label", "int64"),
        ]

        for field_name, expected_dtype in expected_dtypes:
            if field_name not in first_item.dtype.names:
                raise ValueError(f"Field '{field_name}' not found in the dataset.")

            actual_dtype = first_item[field_name].dtype
            if not np.issubdtype(actual_dtype, np.dtype(expected_dtype)):
                raise ValueError(f"Dtype mismatch for '{field_name}': expected {expected_dtype}, got {actual_dtype}")

        # New checks
        num_samples = len(dataset)
        num_bands = len(first_item["bands"])
        pixel_shape = first_item["pixels"].shape
        platform = first_item["platform"]
        gsd = first_item["gsd"]
        bands = first_item["bands"]

        for i in range(len(dataset)):
            item = dataset[i]

            # Check consistency of platform, gsd, and bands
            if item["platform"] != platform:
                raise ValueError(f"Inconsistent platform in sample {i}")
            if item["gsd"] != gsd:
                raise ValueError(f"Inconsistent gsd in sample {i}")
            if not np.array_equal(item["bands"], bands):
                raise ValueError(f"Inconsistent bands in sample {i}")

            # Check pixel shape
            if item["pixels"].shape != pixel_shape:
                raise ValueError(f"Pixel shape mismatch in sample {i}")

            # Check that height == width and is divisible by 8
            _, h, w = item["pixels"].shape
            if h != w:
                raise ValueError(f"Height != Width in sample {i}: {h} != {w}")
            if h % 8 != 0:
                raise ValueError(f"Height (and Width) not divisible by 8 in sample {i}: {h}")

            # Check point shape
            if item["point"].shape != (2,):
                raise ValueError(f"Point shape mismatch in sample {i}")

            # # Validate timestamp (just checking if it's a valid Unix timestamp)
            # try:
            #     datetime.fromtimestamp(item["timestamp"])
            # except ValueError:
            #     raise ValueError(f"Invalid timestamp in sample {i}")

        print("Validation successful!")
        print(f"Number of samples: {num_samples}")
        print(f"Number of bands: {num_bands}")
        print(f"Pixel shape: {pixel_shape}")
        print(f"Platform: {platform.decode('ascii')}")
        print(f"GSD: {gsd}")
        print(f"Bands: {[b.decode('ascii') for b in bands]}")


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
