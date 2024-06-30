"""
It is an example file containing couple of Malevich
objects that consitute an app.

See the documentation for more information:

https://malevichai.github.io/malevich/Apps/Building.html

"""

from typing import Any, List

import numpy as np
import pandas as pd
import stackstac
import torch
from malevich.square import DF, Context, init, processor, scheme

from clay.model import get_encoder
from clay.utils import InputEarthData, get_mock_data, stack_to_datacube


@scheme()
class MalevichInputEarthData(InputEarthData):
    pass


@init(prepare=True)
def init_model(context: Context):
    context.model = get_encoder()


@processor()
def get_malevich_mock_data() -> MalevichInputEarthData:
    data = get_mock_data()[1]
    malevich_data = MalevichInputEarthData(**data.__dict__)
    return malevich_data


@processor()
def inference(inputs: MalevichInputEarthData, context: Context):

    stack = stackstac.stack(
        inputs.items,
        bounds=inputs.bounds,
        snap_bounds=inputs.snap_bounds,
        epsg=inputs.epsg,
        resolution=inputs.resolution,
        # dtype=inputs.dtype,
        rescale=inputs.rescale,
        fill_value=inputs.fill_value,
        assets=inputs.assets,
        resampling=inputs.resampling,
    )
    stack = stack.compute()
    datacube = stack_to_datacube(stack, inputs.lat, inputs.long)
    with torch.no_grad():
        unmsk_patch, unmsk_idx, msk_idx, msk_matrix = context.model(datacube)
    # embeddings = unmsk_patch[:, 0, :].cpu().numpy()
    embeddings = unmsk_patch[:, 0, :].cpu().tolist()  # 1x768
    return embeddings, coordinate_pairs
