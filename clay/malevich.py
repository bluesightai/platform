"""
Core logic that is executed on Malevich cloud
"""


import datetime
from typing import Any

import pandas as pd

from app.schemas.clay import Image
from clay.model import get_encoder, get_embeddings_img
from malevich import square


@square.init(prepare=True)
def prepare_model(context: square.Context):
    """Initialization function. Creates encoder that is shared among runs"""
    context.common = get_encoder()
    
    
@square.scheme()
class SerializedGetEmbeddingInput:
    platform: str
    pixels: Any             # List[List[List[float]]],
    bands: Any              # List[str],
    point: Any              # Tuple[float, float],
    timestamp: int          # datetime
    gsd: float
    


def get_input_from_serialized(serialized: SerializedGetEmbeddingInput):
    """Converts serialized representations back to model"""
    return Image(
        platform=serialized.platform,
        pixels=eval(serialized.pixels.replace(' ', ',')),
        bands=serialized.bands[1:-1].split(' '),
        point=eval(serialized.point.replace(' ', ',')),
        timestamp=serialized.timestamp,
        gsd=serialized.gsd
    )
    
    
@square.processor()
def get_embedding_processor(data: square.DF[SerializedGetEmbeddingInput], context: square.Context):
    """Proxy to get_embeddings_img function"""
    
    # Get the input
    input_ = [
        get_input_from_serialized(SerializedGetEmbeddingInput(**row))
        for row in data.to_dict(orient='records')
    ]
    
    # Construct arguments
    points = [i.point for i in input_]
    bands = input_[0].bands
    pixels = [i.pixels for i in input_]
    datetimes = [datetime.datetime.fromtimestamp(i.timestamp) for i in input_]
    gsd = input_[0].gsd
    platform = input_[0].platform
    encoder = context.common
   
    # Proxy call
    outputs = get_embeddings_img(
        encoder=encoder,
        platform=platform,
        bands=bands,
        pixels=pixels,
        points=points,
        datetimes=datetimes,
        gsd=gsd
    )
    
    # Serialization
    serialized_outputs = [
        repr(o)
        for o in outputs.tolist()
    ]
    return pd.DataFrame(serialized_outputs, columns=['embedding'])

