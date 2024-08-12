from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


class Hyperparameters(BaseModel):
    num_classes: Optional[int] = None
    """The number of classes in the dataset. Specify only for segmentation task."""

    model_config = ConfigDict(use_attribute_docstrings=True, json_schema_extra={"examples": [{"num_classes": 3}]})


class TrainingJobCreate(BaseModel):
    task: Literal["classification", "segmentation"]
    """The task type, which can be either `classification` or `segmentation`."""

    training_file: str
    """The ID of an uploaded file that contains training data.

    See [upload file](https://docs.bluesight.ai/api-reference/files/upload-file) for how to upload a file.
    """

    validation_file: Optional[str] = None
    """The ID of an uploaded file that contains validation data.

    If you provide this file, the data is used to generate validation metrics periodically during fine-tuning. These metrics can be viewed in the fine-tuning results file. The same data should not be present in both train and validation files.
    """

    hyperparameters: Optional[Hyperparameters] = None
    """The hyperparameters used for the training job."""

    model_config = ConfigDict(
        use_attribute_docstrings=True,
        json_schema_extra={
            "examples": [
                {
                    "task": "classification",
                    "training_file": "file-lw3zjxrg",
                    "validation_file": None,
                    "hyperparameters": None,
                }
            ]
        },
    )


class TrainingJob(TrainingJobCreate):
    id: str
    """The object identifier, which can be referenced in the API endpoints."""

    created_at: int
    """The Unix timestamp (in seconds) for when the training job was created."""

    status: Literal[
        "initializing", "downloading_files", "validating_files", "queued", "running", "succeeded", "failed", "cancelled"
    ]
    """
    The current status of the training job, which can be either
    `initializing`, `downloading_files`, `validating_files`, `queued`, `running`, `succeeded`, `failed`, or `cancelled`.
    """

    error: Optional[str] = None
    """
    For training jobs that have `failed`, this will contain more information on
    the cause of the failure.
    """

    trained_model: Optional[str] = None
    """The name of the trained model that is being created.

    The value will be null if the training job is still running.
    """

    finished_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the training job was finished.

    The value will be null if the training job is still running.
    """

    model_config = ConfigDict(
        use_attribute_docstrings=True,
        json_schema_extra={
            "examples": [
                {
                    "task": "classification",
                    "training_file": "file-lw3zjxrg",
                    "validation_file": None,
                    "hyperparameters": None,
                    "id": "trainingjob-g0mos7xr",
                    "created_at": 1723402651,
                    "status": "succeeded",
                    "error": None,
                    "trained_model": "model-3b05uri7",
                    "finished_at": 1723402657,
                }
            ]
        },
    )


class TrainingJobUpdate(BaseModel):
    status: Optional[
        Literal[
            "initializing",
            "downloading_files",
            "validating_files",
            "queued",
            "running",
            "succeeded",
            "failed",
            "cancelled",
        ]
    ] = None
    """
    The current status of the training job, which can be either
    `initializing`, `downloading_files`, `validating_files`, `queued`, `running`, `succeeded`, `failed`, or `cancelled`.
    """

    error: Optional[str] = None
    """
    For training jobs that have `failed`, this will contain more information on
    the cause of the failure.
    """

    trained_model: Optional[str] = None
    """The name of the trained model that is being created.

    The value will be null if the training job is still running.
    """

    finished_at: Optional[float] = None
    """The Unix timestamp (in seconds) for when the training job was finished.

    The value will be null if the training job is still running.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)
