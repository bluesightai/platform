from loguru import logger
from malevich.square import Context

from clay.malevich.bindings import get_malevich_mock_data, inference, init_model

context = Context()

init_model(context)

inputs = get_malevich_mock_data()
embeddings = inference(inputs, context)
logger.info(embeddings)
