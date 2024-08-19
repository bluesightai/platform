import os
import logging
import math
import json
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from PIL import Image
from pyproj import Transformer
from shapely.geometry import box
from tqdm import tqdm
from rasterio.windows import Window

from scripts.embeddings import get_embeddings_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BATCH_SIZE = 100

def process_chip(src, x, y, chip_id, chip_size, output_dir):
    window = rasterio.windows.Window(x, y, chip_size, chip_size)
    chip_data = src.read(window=window)


    # Check if the chip is empty (all zeros or ones)
    if np.all(chip_data <= 1):
        return None

    # Check if the chip is the correct size
    if chip_data.shape[1] < chip_size or chip_data.shape[2] < chip_size:
        print(f"Skipping non-full chip at x={x}, y={y}")
        return None

    chip_filename = os.path.join(output_dir, f"chip_{chip_id}.png")
    Image.fromarray(np.transpose(chip_data, (1, 2, 0))).save(chip_filename)

    # Get the world coordinates of the chip
    world_left, world_top = src.xy(y, x)
    world_right, world_bottom = src.xy(y + chip_size, x + chip_size)

    # Create bounding box geometry
    bbox_geometry = box(world_left, world_bottom, world_right, world_top)

    return {
        "chip_id": chip_id,
        "file_path": chip_filename,
        "geometry": bbox_geometry,
        "image_data": chip_data  # Return the original 8-bit data for embedding
    }

def process_image(tiff_file, output_dir, parquet_dir):
    image_id = os.path.splitext(os.path.basename(tiff_file))[0]
    print(f"Processing image: {image_id}")

    # Create output directories
    image_dir = os.path.join(output_dir, image_id)
    chips_dir = os.path.join(image_dir, "chips")
    os.makedirs(chips_dir, exist_ok=True)

    processed_items = []
    batch_images = []

    try:
        with rasterio.open(tiff_file) as src:
            print(f"Image shape: {src.shape}")
            print(f"Number of bands: {src.count}")
            print(f"Data type: {src.dtypes[0]}")

            crs = src.crs
            print(f"Coordinate Reference System (CRS): {crs}")
            print(f"CRS WKT: {crs.to_wkt()}")
            
            if crs.is_epsg_code:
                print(f"EPSG Code: {crs.to_epsg()}")

            height, width = src.height, src.width
            transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)

            num_chips_x, num_chips_y = math.ceil(width / 224), math.ceil(height / 224)
            total_chips = num_chips_x * num_chips_y

            with tqdm(total=total_chips, desc=f"Processing chips for {image_id}", unit="chip") as pbar:
                for y in range(0, height, 224):
                    for x in range(0, width, 224):
                        try:
                            chip_data = process_chip(
                                src, x, y, len(processed_items), 224, chips_dir
                            )
                            
                            if chip_data is None:  # Skip empty chips
                                pbar.update(1)
                                continue

                            center_x, center_y = x + 112, y + 112
                            center_x_proj, center_y_proj = src.xy(center_y, center_x)
                            lon, lat = transformer.transform(center_x_proj, center_y_proj)

                            chip_data.update(
                                {
                                    "center_lat": lat,
                                    "center_lon": lon,
                                    "image_id": image_id,
                                }
                            )

                            batch_images.append(chip_data["image_data"])
                            del chip_data["image_data"]  # Remove image data to save memory
                            processed_items.append(chip_data)

                            if len(batch_images) == BATCH_SIZE:
                                embeddings = get_embeddings_batch(batch_images)
                                for item, embedding in zip(processed_items[-BATCH_SIZE:], embeddings):
                                    item["embedding"] = embedding
                                batch_images = []

                            pbar.update(1)
                            pbar.set_postfix({"Last chip": len(processed_items)})
                        except Exception as e:
                            logging.error(f"Error processing chip at x={x}, y={y}: {str(e)}")
                            continue  # Skip this chip and continue with the next one

        # Process any remaining images in the batch
        if batch_images:
            embeddings = get_embeddings_batch(batch_images)
            for item, embedding in zip(processed_items[-len(batch_images):], embeddings):
                item["embedding"] = embedding

        print(f"Processed {len(processed_items)} chips from image {image_id}")

        # Create GeoDataFrame and save to parquet
        if processed_items:
            gdf = gpd.GeoDataFrame(processed_items, crs=src.crs)
            gdf_wgs84 = gdf.to_crs(epsg=4326)

            df = pd.DataFrame(
                {
                    "embedding": gdf_wgs84["embedding"],
                    "chip_id": gdf_wgs84["chip_id"],
                    "file_path": gdf_wgs84["file_path"],
                    "image_id": gdf_wgs84["image_id"],
                    "geometry": gdf_wgs84["geometry"].apply(lambda geom: geom.wkb),
                    "center_lat": gdf_wgs84["center_lat"],
                    "center_lon": gdf_wgs84["center_lon"],
                }
            )

            parquet_file = os.path.join(parquet_dir, f"{image_id}_chips.parquet")
            df.to_parquet(parquet_file, engine="pyarrow")

            print(f"Saved chip information to '{parquet_file}'")
        else:
            print(f"No chips were successfully processed for image {image_id}")

        return len(processed_items)
    except Exception as e:
        logging.error(f"Error processing image {image_id}: {str(e)}")
        return 0

# Main execution
input_dir = "./sat_data/color/"  # Replace with the path to your TIFF files
output_dir = "image_chips"
parquet_dir = "parquet_files"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(parquet_dir, exist_ok=True)

# Get the list of TIFF files
tiff_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]

total_chips = 0
errors = []
with tqdm(tiff_files, desc="Processing images", unit="image") as pbar:
    for tiff_file in pbar:
        try:
            full_path = os.path.join(input_dir, tiff_file)
            chips_processed = process_image(full_path, output_dir, parquet_dir)
            total_chips += chips_processed
            print(f"Processed a total of {total_chips} chips across {pbar.n} images.")

            pbar.set_postfix({"Total chips": total_chips, "Last image chips": chips_processed})
        except Exception as e:
            error_msg = f"Error processing image {tiff_file}: {str(e)}"
            logging.error(error_msg)
            errors.append(error_msg)

if errors:
    print(f"Encountered {len(errors)} errors:")
    for error in errors:
        print(error)