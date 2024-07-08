# from malevich import Core, flow
from malevich import Space, SpaceInterpreter, collection, flow, table
from malevich.clay import get_malevich_mock_data, inference, mock_inference


@flow(reverse_id="inference")
def inference_flow():
    inputs = get_malevich_mock_data()
    embeddings = inference(inputs)
    return embeddings


@flow
def inference_flow():
    satelite_images = collection(
        "clay_satilite_images",
        # df=table(...),   # TODO: Define the input structure
        df=table([1, 2, 3], columns=["test"]),
        alias="input_images",
    )

    # TODO: Implement and replace the mock
    # embeddings = inference()

    embeddings = mock_inference(satelite_images)

    # TODO: Save embeddings with Qdrant
    return embeddings


# task = Core(
#     inference_flow,
#     core_host="https://core.malevich.ai/",
#     user="c148403e41894c74a2d83f0703d1223c",
#     access_key="40f191df-d8ce-483a-a5ce-8a654c1f19bd",
# )


space_interpreter = SpaceInterpreter()

task = inference_flow()

task.interpret(space_interpreter)

task.get_interpreted_task().upload()


# space_interpreter.interpret(task)

# task = Space(inference_flow, reverse_id="inference_flow")

# task.configure()
# task.prepare()
# task.run()
# print(task.results()[0].get_df())

# task.upload()

# task.stop()
