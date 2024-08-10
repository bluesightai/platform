import os
import rasterio
from pystac_client import Client
import planetary_computer as pc
import geopandas as gpd
import requests
from shapely.geometry import box
from PIL import Image
import pandas as pd
import numpy as np
import math
from pyproj import Transformer
import json
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Connect to the Planetary Computer STAC API
catalog = Client.open("https://earth-search.aws.element84.com/v1")

# Define your area of interest
aoi = {
    "type": "Polygon",
    "coordinates": [
        [
            [-122.54084201545855, 37.68495197750744],
            [-122.29057106650372, 37.68495197750744],
            [-122.29057106650372, 37.813643519362685],
            [-122.54084201545855, 37.813643519362685],
            [-122.54084201545855, 37.68495197750744]
        ]
    ]
}

# Search for NAIP imagery
search = catalog.search(
    collections=["naip"],  # Specify the collection
    intersects=aoi  # Use the intersects parameter for spatial search
)



BATCH_SIZE = 100

def ensure_rgb(image):
    """Ensure the image is RGB."""
    if image.shape[0] == 4:  # If RGBA, drop the alpha channel
        return image[:3, :, :]
    return image

def resize_image(image, target_size=(224, 224)):
    """Resize image to target size."""
    if image.shape[1:] != target_size:
        image = np.transpose(image, (1, 2, 0))
        image = Image.fromarray(image)
        image = image.resize(target_size, Image.BILINEAR)
        image = np.array(image).transpose(2, 0, 1)
    return image

def get_embeddings_batch(images, gsd=0.6):
    """Get embeddings for a batch of images using the bluesight.ai API."""
    url = "https://api.bluesight.ai/embeddings/img"
    
    payload = {
        "model": "clip",
        "images": [
            {
                "gsd": gsd,
                "bands": ["red", "green", "blue"],
                "pixels": image.tolist()
            }
            for image in images
        ]
    }
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return json.loads(response.text)['embeddings']
    else:
        print(f"Error getting embeddings: {response.text}")
        return [None] * len(images)

def process_chip(src, x, y, chip_id, chip_size, output_dir):
    window = rasterio.windows.Window(x, y, chip_size, chip_size)
    chip_data = src.read(window=window)
    
    chip_data = ensure_rgb(chip_data)
    chip_data = resize_image(chip_data)
    
    # Save the chip
    chip_filename = os.path.join(output_dir, f'chip_{chip_id}.png')
    Image.fromarray(np.transpose(chip_data, (1, 2, 0))).save(chip_filename)
    
    # Get the world coordinates of the chip
    world_left, world_top = src.xy(y, x)
    world_right, world_bottom = src.xy(y + chip_size, x + chip_size)
    
    # Create bounding box geometry
    bbox_geometry = box(world_left, world_bottom, world_right, world_top)
    
    return {
        'chip_id': chip_id,
        'file_path': chip_filename,
        'geometry': bbox_geometry,
        'image_data': chip_data
    }


def process_image(item, output_dir, parquet_dir):
    signed_item = pc.sign(item)
    
    print(f"Processing item: {signed_item.id}")
    
    # Create output directories
    image_dir = os.path.join(output_dir, signed_item.id)
    chips_dir = os.path.join(image_dir, 'chips')
    os.makedirs(chips_dir, exist_ok=True)
    
    processed_items = []
    batch_images = []
    
    try:
        with rasterio.open(signed_item.assets['image'].href) as src:
            # # Print information about the channels
            # print(f"Number of channels: {src.count}")
            # print(f"Channel names: {src.descriptions}")
            # print(f"Channel data types: {[dt for dt in src.dtypes]}")
            # print("Note: NAIP imagery typically doesn't use colormaps.")
            
            height, width = src.height, src.width
            transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)

            num_chips_x, num_chips_y = math.ceil(width / 224), math.ceil(height / 224)
            total_chips = num_chips_x * num_chips_y
            
            with tqdm(total=total_chips, desc=f"Processing chips for {signed_item.id}", unit="chip") as pbar:
                for y in range(num_chips_y):
                    for x in range(num_chips_x):
                        start_x, start_y = x * 224, y * 224
                        current_chip_size = min(224, width - start_x, height - start_y)

                        if current_chip_size > 0:
                            try:
                                chip_data = process_chip(src, start_x, start_y, len(processed_items), current_chip_size, chips_dir)
                                
                                # Print information about the first chip
                                if len(processed_items) == 0:
                                    print(f"First chip shape: {chip_data['image_data'].shape}")
                                    print(f"First chip data type: {chip_data['image_data'].dtype}")
                                    print(f"Channel order when saving: RGB (Red: 0, Green: 1, Blue: 2)")
                                
                                center_x, center_y = start_x + current_chip_size / 2, start_y + current_chip_size / 2
                                center_x_proj, center_y_proj = src.xy(center_y, center_x)
                                lon, lat = transformer.transform(center_x_proj, center_y_proj)
                                
                                chip_data.update({
                                    'center_lat': lat,
                                    'center_lon': lon,
                                    'image_id': signed_item.id,
                                })
                                
                                batch_images.append(chip_data['image_data'])
                                del chip_data['image_data']  # Remove image data to save memory
                                processed_items.append(chip_data)
                                
                                if len(batch_images) == BATCH_SIZE:
                                    embeddings = get_embeddings_batch(batch_images)
                                    for item, embedding in zip(processed_items[-BATCH_SIZE:], embeddings):
                                        item['embedding'] = embedding
                                    batch_images = []
                                
                                pbar.update(1)
                                pbar.set_postfix({"Last chip": len(processed_items)})
                            except rasterio.errors.RasterioIOError as e:
                                logging.error(f"Error processing chip at x={start_x}, y={start_y}: {str(e)}")
                                continue  # Skip this chip and continue with the next one

        # Process any remaining images in the batch
        if batch_images:
            embeddings = get_embeddings_batch(batch_images)
            for item, embedding in zip(processed_items[-len(batch_images):], embeddings):
                item['embedding'] = embedding

        print(f"Processed {len(processed_items)} chips from image {signed_item.id}")
        
        # Create GeoDataFrame and save to parquet
        if processed_items:
            gdf = gpd.GeoDataFrame(processed_items, crs=src.crs)
            gdf_wgs84 = gdf.to_crs(epsg=4326)
            
            df = pd.DataFrame({
                'embedding': gdf_wgs84['embedding'],
                'chip_id': gdf_wgs84['chip_id'],
                'file_path': gdf_wgs84['file_path'],
                'image_id': gdf_wgs84['image_id'],
                'geometry': gdf_wgs84['geometry'].apply(lambda geom: geom.wkb),
                'center_lat': gdf_wgs84['center_lat'],
                'center_lon': gdf_wgs84['center_lon']
            })
            
            parquet_file = os.path.join(parquet_dir, f'{signed_item.id}_chips.parquet')
            df.to_parquet(parquet_file, engine='pyarrow')
            
            print(f"Saved chip information to '{parquet_file}'")
        else:
            print(f"No chips were successfully processed for image {signed_item.id}")
        
        return len(processed_items)
    except Exception as e:
        logging.error(f"Error processing image {signed_item.id}: {str(e)}")
        return 0

# Main execution
output_dir = 'image_chips'
parquet_dir = 'parquet_files'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(parquet_dir, exist_ok=True)

# Get the total number of images
total_images = sum(1 for _ in search.items())

total_chips = 0
errors = []
with tqdm(search.items(), total=total_images, desc="Processing images", unit="image") as pbar:
    for i, item in enumerate(pbar):
        try:
            chips_processed = process_image(item, output_dir, parquet_dir)
            total_chips += chips_processed
            print(f"Processed a total of {total_chips} chips across {i+1} images.")

            pbar.set_postfix({"Total chips": total_chips, "Last image chips": chips_processed})
        except Exception as e:
            error_msg = f"Error processing image {item.id}: {str(e)}"
            logging.error(error_msg)
            errors.append(error_msg)

if errors:
    print(f"Encountered {len(errors)} errors:")
    for error in errors:
        print(error)