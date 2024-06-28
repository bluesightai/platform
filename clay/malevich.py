"""
It is an example file containing couple of Malevich
objects that consitute an app.

See the documentation for more information:

https://malevichai.github.io/malevich/Apps/Building.html

"""
import pandas as pd
import torch
from malevich import table
from malevich.square import DF, Context, init, processor, scheme

@scheme()
class InputEarthData:
    # TODO: Define the input earth data
    pass

@scheme()
class ModelConfiguration:
    # TODO: Specify tweakable configurations for each run
    pass
    
@init(prepare=True)
def init_model(context: Context):
    context.model = ... # TODO: Initialize, the model. Only run once and preserved for runs

@processor()
def inference(inputs: DF[InputEarthData], context: Context):
    config: ModelConfiguration = context.app_cfg
    # TODO: Implement clay inference
    pass


@processor()
def mock_inference(inputs: DF[InputEarthData], context: Context):
    name_of_vectors = context.app_cfg.get('name', 'clay_embeddings')
    
    outputs = torch.rand(len(inputs), 768)
    outputs = [repr(o.cpu().tolist()) for o in outputs]
    return pd.DataFrame(outputs, columns=[name_of_vectors])

@processor()
def write_asset_file(just_input: DF, context: Context):
    with open(context.get_object('example_file.txt').path) as f:
        return pd.DataFrame([f.read()], columns=['contents'])