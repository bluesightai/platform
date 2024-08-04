from typing import Literal, Optional

from pydantic import BaseModel


class Hyperparameters(BaseModel):
    pass


class TrainingJobCreate(BaseModel):
    task: Literal["classification", "segmentation"]
    """The task type, which can be either `classification` or `segmentation`."""

    training_file: str
    """The file ID used for training."""

    validation_file: Optional[str] = None
    """The file ID used for validation."""

    hyperparameters: Optional[Hyperparameters] = None
    """The hyperparameters used for the training job."""


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
