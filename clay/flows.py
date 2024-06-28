from malevich.clay import mock_inference, inference, write_asset_file
import pandas as pd

from malevich import Core, collection, flow, table


@flow(reverse_id="inference")
def inference_flow():
    satelite_images = collection(
        "clay_satilite_images",
        # df=table(...),   # TODO: Define the input structure
        df=table([1, 2, 3], columns=['test']),
        alias='input_images'
    )
   
    # TODO: Implement and replace the mock
    # embeddings = inference()
    
    embeddings = mock_inference(satelite_images)

    # TODO: Save embeddings with Qdrant
    return embeddings

@flow
def just_test_asset():
    return write_asset_file(
        collection('test', df=table(['some_data'], columns=['some_column']))) # just mock input to test


# Test asset

task = Core(just_test_asset, core_host='https://core.malevich.ai/')
task.prepare()
task.run()
print(task.results()[0].get_df())
task.stop()


# Test inference

task = Core(inference_flow, core_host='https://core.malevich.ai/')
task.prepare()
task.run()
print(task.results()[0].get_df())
task.stop()
