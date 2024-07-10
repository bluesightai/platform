import csv
import json
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

source_file_path = "./data/dakota_big.pkl"
target_file_path = "./data/dakota_big_computed.csv"
n = 300


def normalize(v):
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norm


# for idx in top_n_closest_indices:
#     (lat_closest, lon_closest), stack, embedding, _ = data[idx]
#     print(
#         f"Visualizing stack for point at (lat: {lat_closest}, lon: {lon_closest}) with similarity: {similarities[idx]}"
#     )
#     visualize_stack(stack=stack)

# Replace 'your_file.pkl' with the path to your .pkl file

with open(source_file_path, "rb") as file:
    data = pickle.load(file)

coordinates = np.array([(lat, lon) for (lat, lon), *_ in data])
bboxes = np.array([(nw_lat, nw_lon, se_lan, se_lon) for *_, (nw_lat, nw_lon, se_lan, se_lon) in data])
embeddings = np.array([item[2] for item in data])
result: List[
    Tuple[
        List[float],
        List[float],
        List[List[float]],
    ]
] = []

for (lat, lon), stack, embedding, (nw_lat, nw_lon, se_lan, se_lon) in tqdm(data):
    target_embedding_norm = embedding / np.linalg.norm(embedding)
    embeddings_norm = normalize(embeddings)
    similarities = np.dot(embeddings_norm, target_embedding_norm)
    top_n_closest_indices = np.argsort(similarities)[-n:][::-1]
    result.append(
        (
            [lat, lon],
            [nw_lat, nw_lon, se_lan, se_lon],
            [
                [
                    coordinates[idx].tolist()[0],
                    coordinates[idx].tolist()[1],
                    float(similarities[idx]),
                    bboxes[idx].tolist()[0],
                    bboxes[idx].tolist()[1],
                    bboxes[idx].tolist()[2],
                    bboxes[idx].tolist()[3],
                ]
                for idx in top_n_closest_indices
            ],
        )
    )


with open(target_file_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["latitude", "longitude", "bbox", "scores"])
    for (lat, lon), bbox, scores in result:
        writer.writerow([lat, lon, bbox, scores])
