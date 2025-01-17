import concurrent.futures
import pickle

import fire
from loguru import logger
from tqdm import tqdm

from clay.config import config
from clay.model import get_embedding
from clay.utils import get_bbox, get_square_centers


def process_center(lat: float, lon: float, size: int, gsd: int, start: str):
    try:
        embedding, stack = get_embedding(lat=lat, lon=lon, size=size, gsd=gsd, start=start)
        return ((lat, lon), stack, embedding.squeeze(), get_bbox(lat=lat, lon=lon, size=size, gsd=gsd))
    except Exception as e:
        logger.error(f"Failed to process ({lat}, {lon}): {e}")
        return None


def build_embeddings(
    nw_lat: float,
    nw_lon: float,
    se_lat: float,
    se_lon: float,
    size: int,
    gsd: int,
    shift_percentage: float,
    start: str,
    max_items_per_file: int = 5000,
):
    """
    north dakota
    python scripts/build_embeddings.py 48.43996086356826 -101.38773653662064 48.3931714959406 -101.30094856808536 128 0.6 1.0 '2022-01-01'

    nevada
    python scripts/build_embeddings.py 36.25079152886272 -115.05193360689421 36.21906455965353, -115.01519546690653 128 0.6 1.0 '2022-01-01'
    """

    base_filename = f"{nw_lat}:{nw_lon}|{se_lat}:{se_lon}|{config.platform}|{size}|{shift_percentage}|{gsd}.pkl"
    pixel_shift = int(shift_percentage * size)

    centers = get_square_centers(
        nw_lat=nw_lat, nw_lon=nw_lon, se_lat=se_lat, se_lon=se_lon, pixel_shift=pixel_shift, gsd=gsd
    )
    logger.info(f"Will be creating {len(centers)} points!")

    data, idx = [], 0
    with tqdm(total=len(centers)) as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(process_center, lat=lat, lon=lon, size=size, gsd=gsd, start=start): (lat, lon)
                for lat, lon in centers
            }

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    data.append(result)
                pbar.update(1)

                if len(data) >= max_items_per_file:
                    with open(f"{idx}|{base_filename}", "wb") as f:
                        pickle.dump(data, f)
                    data.clear()
                    idx += 1
                    logger.info(f"Exported to file: {f'{idx}|{base_filename}'}")

    logger.info(f"Got {len(data)} points!")

    with open(f"{idx}|{base_filename}", "wb") as f:
        pickle.dump(data, f)

    logger.info(f"Exported to file: {f'{idx}|{base_filename}'}")


if __name__ == "__main__":
    fire.Fire(build_embeddings)
