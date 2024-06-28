"""
It is an example file containing couple of Malevich
objects that consitute an app.

See the documentation for more information:

https://malevichai.github.io/malevich/Apps/Building.html

"""

from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import stackstac
import torch
from malevich import table
from malevich.square import DF, Context, init, processor, scheme
from model import get_encoder
from rasterio.enums import Resampling
from stackstac.stack import Bbox
from utils import stack_to_datacube
from xarray import DataArray


@scheme()
class InputEarthData:
    items: List[Any]
    bounds: Bbox
    snap_bounds: bool
    epsg: int
    resolution: int
    dtype: np.dtype
    rescale: bool
    fill_value: int | float
    assets: List[str]
    resampling: Resampling
    lat: float
    long: float


# @scheme()
# class ModelConfiguration:
#     # TODO: Specify tweakable configurations for each run
#     pass
#
#
@init(prepare=True)
def init_model(context: Context):
    context.model = get_encoder()


@processor()
def inference(inputs: InputEarthData, context: Context):
    # config: ModelConfiguration = context.app_cfg

    stack = stackstac.stack(
        inputs.items,
        bounds=inputs.bounds,
        snap_bounds=inputs.snap_bounds,
        epsg=inputs.epsg,
        resolution=inputs.resolution,
        dtype=inputs.dtype,
        rescale=inputs.rescale,
        fill_value=inputs.fill_value,
        assets=inputs.assets,
        resampling=inputs.resampling,
    )
    datacube = stack_to_datacube(stack, inputs.lat, inputs.long)
    with torch.no_grad():
        unmsk_patch, unmsk_idx, msk_idx, msk_matrix = context.model(datacube)
    embeddings = unmsk_patch[:, 0, :].cpu().numpy()
    return embeddings


@processor()
def mock_inference(inputs: DF[InputEarthData], context: Context):
    name_of_vectors = context.app_cfg.get("name", "clay_embeddings")

    outputs = torch.rand(len(inputs), 768)
    outputs = [repr(o.cpu().tolist()) for o in outputs]
    return pd.DataFrame(outputs, columns=[name_of_vectors])


@processor()
def write_asset_file(just_input: DF, context: Context):
    with open(context.get_object("example_file.txt").path) as f:
        return pd.DataFrame([f.read()], columns=["contents"])

