from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import lightning as L
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from loguru import logger
from numpy._typing import NDArray
from sklearn import svm
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torchmetrics.classification import F1Score, MulticlassJaccardIndex

from clay.model import Segmentor
from clay.utils import get_datacube, get_stats


def train_classification(embeddings: NDArray, labels: NDArray) -> Tuple[svm.SVC, Dict[str, Any]]:

    # from xgboost import XGBClassifier
    # from sklearn.model_selection import GridSearchCV
    # param_grid = {
    #    'n_estimators': [50, 100, 150],
    #    'learning_rate': [0.01, 0.1, 0.2],
    #    'max_depth': [3, 5, 7]
    # }
    # xgb = XGBClassifier()
    # grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=2, n_jobs=-1, verbose=1)
    # grid_search.fit(embeddings, labels)
    # best_params = grid_search.best_params_
    # best_model = grid_search.best_estimator_

    # print(f"Best parameters found: {best_params}")

    clf = svm.SVC()
    clf.fit(embeddings, labels)

    details = {k: v for k, v in clf.__dict__.items() if k not in ["support_vectors_"]}
    details = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in details.items()}

    return clf, details


def predict_classification(clf: svm.SVC, embeddings: NDArray) -> NDArray:
    labels = clf.predict(embeddings)
    return labels


class SegmentationDataset(Dataset):
    """
    Dataset class for the Chesapeake Bay segmentation dataset.
    """

    def __init__(
        self,
        file_path: Path,
    ):

        self.file_path = file_path
        self.f = h5py.File(file_path, "r")
        self.dataset: h5py.Dataset = self.f["data"]

        # with h5py.File(file_path, "r") as f:
        # dataset: h5py.Dataset = f["data"]
        first_item: Dataset = self.dataset[0]

        self.length = len(self.dataset)
        self.gsd = first_item["gsd"].item()
        self.bands = [band.decode("ascii") for band in first_item["bands"]]
        self.platform = first_item["platform"].decode("ascii")
        self.stats = get_stats(bands=self.bands, pixels=first_item["pixels"], platform=self.platform, wavelengths=None)

        logger.debug(f"Found dataset of {self.length} samples.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        # with h5py.File(self.file_path, "r") as f:

        # dataset: h5py.Dataset = f["data"]

        datacube = get_datacube(
            gsd=self.gsd,
            stats=self.stats,
            pixels=[self.dataset[idx]["pixels"]],
            datetimes=[datetime.utcfromtimestamp(self.dataset[idx]["timestamp"])],
            points=[self.dataset[idx]["point"].tolist()],
        )

        datacube["pixels"] = datacube["pixels"][0]
        datacube["time"] = datacube["time"][0]
        datacube["latlon"] = datacube["latlon"][0]
        datacube["label"] = torch.from_numpy(self.dataset[idx]["label"])

        return datacube


class SegmentationDataModule(L.LightningDataModule):

    def __init__(  # noqa: PLR0913
        self,
        train_file_path: Path,
        validation_file_path: Path | None,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.train_file_path = train_file_path
        self.validation_file_path = validation_file_path

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """
        Setup datasets for training and validation.

        Args:
            stage (str): Stage identifier ('fit' or 'test').
        """
        if stage in {"fit", None}:
            self.trn_ds = SegmentationDataset(
                file_path=self.train_file_path,
            )
            self.val_ds = None
            if self.validation_file_path:
                self.val_ds = SegmentationDataset(
                    file_path=self.validation_file_path,
                )

    def train_dataloader(self):
        return DataLoader(
            self.trn_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=CustomBatchCollator(),
        )

    def val_dataloader(self):
        if not self.val_ds:
            return None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=CustomBatchCollator(),
        )


class CustomBatchCollator:
    def __call__(self, batch):
        # Extract the elements we want to stack
        pixels = [item["pixels"] for item in batch]
        times = [item["time"] for item in batch]
        latlons = [item["latlon"] for item in batch]

        # Stack these elements
        stacked_pixels = default_collate(pixels)
        stacked_times = default_collate(times)
        stacked_latlons = default_collate(latlons)

        # Create the output dictionary
        output = {
            "platform": batch[0]["platform"],
            "gsd": batch[0]["gsd"],
            "waves": batch[0]["waves"],
            "pixels": stacked_pixels,
            "time": stacked_times,
            "latlon": stacked_latlons,
        }

        if "label" in batch[0]:
            labels = [item["label"] for item in batch]
            stacked_labels = default_collate(labels)
            output |= {"label": stacked_labels}

        return output


class SegmentationDatasetInference(Dataset):
    """
    Dataset class for the Chesapeake Bay segmentation dataset.

    Args:
        chip_dir (str): Directory containing the image chips.
        label_dir (str): Directory containing the labels.
        metadata (Box): Metadata for normalization and other dataset-specific details.
        platform (str): Platform identifier used in metadata.
    """

    def __init__(
        self,
        gsd: float,
        bands: List[str],
        pixels: List[List[List[List[float]]]],  # [B, C, H, W]
        labels: List[List[List[int]]] | None,  # [B, H, W]
        platform: str | None,
        wavelengths: List[float] | None,
        points: List[Tuple[float, float] | None],
        datetimes: List[datetime | None],
    ):

        logger.debug(f"Creating dataset of {len(pixels)} samples...")
        self.gsd = gsd
        self.stats = get_stats(bands=bands, pixels=pixels, platform=platform, wavelengths=wavelengths)
        self.pixels = pixels
        self.datetimes = datetimes
        self.points = points
        self.labels = labels

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):

        datacube = get_datacube(
            gsd=self.gsd,
            stats=self.stats,
            pixels=[self.pixels[idx]],
            datetimes=[self.datetimes[idx]],
            points=[self.points[idx]],
        )

        datacube["pixels"] = datacube["pixels"][0]
        datacube["time"] = datacube["time"][0]
        datacube["latlon"] = datacube["latlon"][0]

        if self.labels:
            datacube |= {"label": torch.from_numpy(np.array(self.labels[idx]))}

        return datacube


class SegmentationDataModuleInference(L.LightningDataModule):

    def __init__(  # noqa: PLR0913
        self,
        gsd: float,
        bands: List[str],
        pixels: List[List[List[List[float]]]],  # [B, C, H, W]
        labels: List[List[List[int]]] | None,  # [B, H, W]
        platform: str | None,
        wavelengths: List[float] | None,
        points: List[Tuple[float, float] | None],
        datetimes: List[datetime | None],
        batch_size: int,
        num_workers: int,
        train_test_ratio: float = 0.8,
    ):
        super().__init__()
        self.gsd = gsd
        self.bands = bands
        self.pixels = pixels
        self.labels = labels
        self.platform = platform
        self.wavelengths = wavelengths
        self.points = points
        self.datetimes = datetimes
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.n_train_samples = int(train_test_ratio * len(pixels))

    def setup(self, stage=None):
        """
        Setup datasets for training and validation.

        Args:
            stage (str): Stage identifier ('fit' or 'test').
        """
        if stage in {"fit", None}:
            self.trn_ds = SegmentationDatasetInference(
                gsd=self.gsd,
                bands=self.bands,
                pixels=self.pixels[: self.n_train_samples],
                platform=self.platform,
                wavelengths=self.wavelengths,
                points=self.points[: self.n_train_samples],
                datetimes=self.datetimes[: self.n_train_samples],
                labels=self.labels[: self.n_train_samples] if self.labels else None,
            )
            self.val_ds = SegmentationDatasetInference(
                gsd=self.gsd,
                bands=self.bands,
                pixels=self.pixels[self.n_train_samples :],
                platform=self.platform,
                wavelengths=self.wavelengths,
                points=self.points[self.n_train_samples :],
                datetimes=self.datetimes[self.n_train_samples :],
                labels=self.labels[self.n_train_samples :] if self.labels else None,
            )

    def train_dataloader(self):
        return DataLoader(
            self.trn_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=CustomBatchCollator(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=CustomBatchCollator()
        )


class SegmentorTraining(L.LightningModule):
    """
    LightningModule for segmentation tasks, utilizing Clay Segmentor.

    Attributes:
        model (nn.Module): Clay Segmentor model.
        loss_fn (nn.Module): The loss function.
        iou (Metric): Intersection over Union metric.
        f1 (Metric): F1 Score metric.
        lr (float): Learning rate.
    """

    def __init__(self, num_classes, feature_maps, lr, wd, b1, b2):  # # noqa: PLR0913
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for checkpointing
        self.model = Segmentor(num_classes=num_classes, feature_maps=feature_maps)

        self.loss_fn = smp.losses.FocalLoss(mode="multiclass")
        self.iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            average="weighted",
        )
        self.f1 = F1Score(
            task="multiclass",
            num_classes=num_classes,
            average="weighted",
        )

    def forward(self, datacube):
        return self.model(datacube)

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            dict: A dictionary containing the optimizer and scheduler
            configuration.
        """
        optimizer = optim.AdamW(
            [param for name, param in self.model.named_parameters() if param.requires_grad],
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=1000, T_mult=1, eta_min=self.hparams.lr * 100, last_epoch=-1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def shared_step(self, batch, batch_idx, phase):
        """
        Shared step for training and validation.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.
            phase (str): The phase (train or val).

        Returns:
            torch.Tensor: The loss value.
        """
        labels = batch["label"].long()
        outputs = self(batch)
        outputs = F.interpolate(
            outputs, size=(labels.shape[-2], labels.shape[-1]), mode="bilinear", align_corners=False
        )  # Resize to match labels size

        loss = self.loss_fn(outputs, labels)
        iou = self.iou(outputs, labels)
        f1 = self.f1(outputs, labels)

        # Log metrics
        self.log(f"{phase}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{phase}/iou", iou, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{phase}/f1", f1, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        return self.shared_step(batch, batch_idx, "val")
