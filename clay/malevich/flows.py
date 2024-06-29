from malevich import Core, flow
from malevich.clay import get_malevich_mock_data, inference


@flow(reverse_id="inference")
def inference_flow():
    inputs = get_malevich_mock_data()
    embeddings = inference(inputs)
    return embeddings


task = Core(
    inference_flow,
    core_host="https://core.malevich.ai/",
    user="c148403e41894c74a2d83f0703d1223c",
    access_key="40f191df-d8ce-483a-a5ce-8a654c1f19bd",
)
task.prepare()
task.run()
print(task.results()[0].get_df())
task.stop()
