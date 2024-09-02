import json

import requests


def get_embeddings_batch(images, gsd=0.6):
    """Get embeddings for a batch of images using the bluesight.ai API."""
    url = "https://api.bluesight.ai/embeddings/img"

    payload = {
        "model": "clip",
        "images": [{"gsd": gsd, "bands": ["red", "green", "blue"], "pixels": image.tolist()} for image in images],
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        return json.loads(response.text)["embeddings"]
    else:
        print(f"Error getting embeddings: {response.text}")
        return [None] * len(images)
