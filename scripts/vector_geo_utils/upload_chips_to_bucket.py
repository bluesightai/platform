import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv
from typing import List, Dict

# Load environment variables
load_dotenv()


url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")


supabase: Client = create_client(url, key)

# Set the bucket name
BUCKET_NAME = "box_chips"

# Set paths for parquet file and chips directory
PARQUET_FILE_PATH = "../parquet_files/2_chips.parquet"  # Update this path
CHIPS_DIRECTORY = "../image_chips/2/chips"  # Update this path


def ensure_bucket_exists(bucket_name: str) -> None:
    """Ensure that the specified bucket exists, creating it if necessary."""
    try:
        supabase.storage.get_bucket(bucket_name)
    except Exception as e:
        if "Not Found" in str(e):
            print(f"Bucket '{bucket_name}' not found. Creating it...")
            supabase.storage.create_bucket(bucket_name)
            print(f"Bucket '{bucket_name}' created successfully.")
        else:
            raise


def list_files_in_bucket(bucket_name: str) -> List[Dict]:
    """List all files in the specified bucket."""
    try:
        return supabase.storage.from_(bucket_name).list()
    except Exception as e:
        print(f"Error listing files in bucket '{bucket_name}': {e}")
        return []


def upload_image_to_supabase(file_path: str, file_name: str) -> bool:
    """Upload an image to Supabase storage."""
    try:
        with open(file_path, "rb") as file:
            supabase.storage.from_(BUCKET_NAME).upload(
                path=file_name, file=file, file_options={"content-type": "image/png"}
            )
        print(f"Uploaded {file_name} to Supabase")
        return True
    except Exception as e:
        print(f"Error uploading {file_name}: {e}")
        return False


def main():
    # Ensure the bucket exists
    ensure_bucket_exists(BUCKET_NAME)

    # Read the parent file with chip IDs
    try:
        parent_df = pd.read_parquet(PARQUET_FILE_PATH)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return

    # Create a dictionary mapping chip filenames to their IDs
    chip_id_map = {f"chip_{row['chip_id']}.png": str(row["chip_id"]) for _, row in parent_df.iterrows()}

    # Get all PNG files in the chips directory
    try:
        png_files = [f for f in os.listdir(CHIPS_DIRECTORY) if f.endswith(".png")]
    except Exception as e:
        print(f"Error reading chips directory: {e}")
        return

    # Get list of files already in the bucket
    existing_files = list_files_in_bucket(BUCKET_NAME)
    existing_file_names = set(file["name"] for file in existing_files)

    successful_uploads = 0
    failed_uploads = 0

    for png_file in png_files:
        if png_file in chip_id_map:
            chip_id = chip_id_map[png_file]
            file_path = os.path.join(CHIPS_DIRECTORY, png_file)
            new_file_name = f"{chip_id}.png"

            # Check if file already exists in the bucket
            if new_file_name in existing_file_names:
                print(f"File {new_file_name} already exists in the bucket. Skipping...")
                continue

            if upload_image_to_supabase(file_path, new_file_name):
                successful_uploads += 1
            else:
                failed_uploads += 1
        else:
            print(f"Warning: No matching chip ID found for {png_file}")

    print(f"Upload complete. Successful uploads: {successful_uploads}, Failed uploads: {failed_uploads}")


if __name__ == "__main__":
    main()
