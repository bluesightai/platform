import logging
import multiprocessing
import os
import traceback

import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from shapely import wkb
from tqdm import tqdm

from scripts.vector_geo_utils.db import postgres_uri

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def batch_insert_parquet_to_postgres(args):
    parquet_file_path, batch_size = args
    try:
        # Load the DataFrame
        df = pd.read_parquet(parquet_file_path)

        # Select relevant columns
        df = df[["geometry", "embedding"]]

        # Decode the binary geometry using shapely's wkb module
        df["geometry"] = df["geometry"].apply(wkb.loads)

        # Create a list of tuples for batch insert
        data_to_insert = []
        for _, row in df.iterrows():
            # Convert the geometry to WKT format
            wkt_geometry = row["geometry"].wkt
            # Ensure the embedding is in the correct format (list or array)
            embedding = row["embedding"]
            if isinstance(embedding, str):
                embedding = eval(embedding)  # Convert string representation to list
            data_to_insert.append((wkt_geometry, embedding.tolist()))

            # Insert in batches
            if len(data_to_insert) >= batch_size:
                with psycopg2.connect(postgres_uri) as conn:
                    with conn.cursor() as cur:
                        query = """
                        INSERT INTO clip_boxes (location, embedding)
                        VALUES (ST_GeomFromText(%s, 4326)::geography, %s)
                        ON CONFLICT (location) DO NOTHING
                        """
                        execute_batch(cur, query, data_to_insert)
                data_to_insert = []

        # Insert any remaining data
        if data_to_insert:
            with psycopg2.connect(postgres_uri) as conn:
                with conn.cursor() as cur:
                    query = """
                    INSERT INTO clip_boxes (location, embedding)
                    VALUES (ST_GeomFromText(%s, 4326)::geography, %s)
                    ON CONFLICT (location) DO NOTHING
                    """
                    execute_batch(cur, query, data_to_insert)

        return parquet_file_path, None  # Success
    except Exception as e:
        error_msg = f"Error processing {parquet_file_path}: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        return parquet_file_path, error_msg  # Return the error message


def process_parquet_files(directory_path, batch_size=200):
    parquet_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".parquet"):
                parquet_files.append(os.path.join(root, file))

    logging.info(f"Found {len(parquet_files)} parquet files to process")

    # Create a list of arguments for each file
    args_list = [(file, batch_size) for file in parquet_files]

    successful = []
    failed = []

    # Use multiprocessing to process files in parallel
    with multiprocessing.Pool() as pool:
        # Use tqdm to create a progress bar
        for i, (file_path, error) in enumerate(
            tqdm(
                pool.imap_unordered(batch_insert_parquet_to_postgres, args_list),
                total=len(parquet_files),
                desc="Processing files",
            )
        ):
            if error is None:
                successful.append(file_path)
            else:
                failed.append((file_path, error))

            # Log progress every 10 files
            if (i + 1) % 10 == 0 or i == len(parquet_files) - 1:
                logging.info(
                    f"Processed {i+1}/{len(parquet_files)} files. Successful: {len(successful)}, Failed: {len(failed)}"
                )

    # Save failed files to a text file
    with open("failed_files.txt", "w") as f:
        for file_path, error in failed:
            f.write(f"{file_path}\n")
            f.write(f"Error: {error}\n\n")

    logging.info(f"Processing complete. Successful: {len(successful)}, Failed: {len(failed)}")
    logging.info(f"Failed files have been saved to 'failed_files.txt'")


if __name__ == "__main__":
    directory_path = "../../"  # Replace with your directory containing parquet files
    process_parquet_files(directory_path)

