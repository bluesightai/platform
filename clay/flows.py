from malevich import Core, flow
from malevich.clay import inference
from utils import get_mock_data


@flow(reverse_id="inference")
def inference_flow():
    _, inputs = get_mock_data()
    embeddings = inference(inputs)
    return embeddings


task = Core(inference_flow, core_host="https://core.malevich.ai/", user="furiousteabag@gmail.com")
task.prepare()
task.run()
print(task.results()[0].get_df())
task.stop()
