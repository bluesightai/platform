import json
import pickle
import uuid

import geopandas as gpd
import pandas as pd
import pygeohash as pgh
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from shapely import wkt
from shapely.geometry import Polygon
from tqdm import tqdm

blacklist_metadata_key = ["x", "y"]


def generate_square_geo(bbox):
    nw_lat, nw_lon = bbox[0], bbox[1]
    se_lat, se_lon = bbox[2], bbox[3]

    ne_lat, ne_lon = nw_lat, se_lon
    sw_lat, sw_lon = se_lat, nw_lon

    coordinates = [(nw_lon, nw_lat), (ne_lon, ne_lat), (se_lon, se_lat), (sw_lon, sw_lat), (nw_lon, nw_lat)]

    square = Polygon(coordinates)
    return square


def parse_raw(path: str, add_metadata: bool = True) -> gpd.GeoDataFrame:

    with open(path, "rb") as f:
        data = pickle.load(f)

    stack = {}

    all_metadata = []
    wkts = []

    for loc, stack, vector, bbox in tqdm(data):
        loc_id = pgh.encode(*loc)

        square = generate_square_geo(bbox)
        wkts.append(square.wkt)

        metadata = {}

        if add_metadata:
            for key, value in stack.coords.items():
                if key in blacklist_metadata_key:
                    continue
                converted = value.to_dict()
                metadata[key] = converted["data"]

        metadata = {"metadata": metadata, "loc_id": loc_id, "lon": loc[1], "lat": loc[0], "vector": vector}

        all_metadata.append(metadata)

    df = pd.DataFrame(all_metadata)

    doc_df = df.to_json()
    loaded = json.loads(doc_df)

    geometries = [wkt.loads(wkt_str) for wkt_str in wkts]

    gdf = gpd.GeoDataFrame(data=loaded, geometry=geometries)

    return gdf


def export_geojson(df: gpd.GeoDataFrame, path: str):
    expanded_metadata = df["metadata"].apply(pd.Series)

    df_expanded = pd.concat([df, expanded_metadata], axis=1)

    df_expanded.drop(["metadata", "vector"], axis=1, inplace=True)

    grid = df_expanded.to_json()

    with open(path, "w") as f:
        f.write(grid)


def batch_list(input_list, batch_size):
    """
    Splits a 1D list into batches of a specified size.

    Parameters:
    - input_list: List of elements to be batched.
    - batch_size: The size of each batch.

    Returns:
    - A list of lists, where each sublist is a batch of the original list.
    """
    return [input_list[i : i + batch_size] for i in range(0, len(input_list), batch_size)]


def index(
    raw_path: str,
    index_name: str,
    url: str = "https://qdrant.malevich.ai/",
    vector_size: int = 768,
    add_metadata: bool = False,
    index: bool = False,
    export_path: str | None = None,
    recreate_collection: bool = False,
    batch_size: int = 200,
):
    client = QdrantClient(url=url, port=443, timeout=10000)

    if recreate_collection:
        client.recreate_collection(
            collection_name=index_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    gdf = parse_raw(raw_path, add_metadata)

    row_batches = batch_list([row for i, row in gdf.iterrows()], batch_size)

    if index:
        for row_batch in tqdm(row_batches):
            client.upsert(
                collection_name=index_name,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=row["vector"],
                        payload={
                            "location": {"lon": row["lon"], "lat": row["lat"]},
                            "geometry": row["geometry"].wkt,
                            "metadata": row["metadata"],
                        },
                    )
                    for row in row_batch
                ],
                wait=False,
            )
    if export_path:
        export_geojson(gdf, export_path)


if __name__ == "__main__":
    index(
        "./data/dakota.pkl",
        "greenvision_military_bases_dakota",
        add_metadata=True,
        export_path="greenvision_military_bases_dakota.json",
        index=True,
        recreate_collection=True,
    )
