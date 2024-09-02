import psycopg2
import logging
from typing import List, Tuple
from psycopg2.extensions import AsIs
from psycopg2.extras import execute_values
from psycopg2.extras import execute_batch
from scripts.vector_geo_utils import db
from tqdm import tqdm
from db import postgres_uri

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split a list into chunks of specified size."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def process_given_region(area_id: int) -> int:
    total_updated = 0
    try:
        with psycopg2.connect(postgres_uri) as conn:
            with conn.cursor() as cur:
                # Get the list of intersecting search_box IDs
                cur.execute("SELECT array_agg(box_id) FROM get_intersecting_search_boxes(%s);", (area_id,))
                result = cur.fetchone()

                if result and result[0]:
                    box_ids = result[0]
                    logging.info(f"Fetched {len(box_ids)} intersecting search box IDs for area {area_id}")

                    # Process in batches of 50
                    batches = chunk_list(box_ids, 1000)
                    update_query = """
                        UPDATE search_boxes
                        SET search_area_id = %s
                        WHERE id = ANY(%s)
                    """

                    # Use tqdm for progress tracking
                    for batch in tqdm(batches, desc=f"Processing area {area_id}"):
                        cur.execute(update_query, (area_id, batch))
                        conn.commit()  # Commit after each batch

                    # Get the total number of updated rows
                    cur.execute(
                        "SELECT COUNT(*) FROM search_boxes WHERE search_area_id = %s AND id = ANY(%s)",
                        (area_id, box_ids),
                    )
                    total_updated = cur.fetchone()[0]

                    logging.info(f"Updated {total_updated} search boxes for area {area_id}")

                    return total_updated
                else:
                    logging.info(f"No intersecting search boxes found for area {area_id}")
                    return 0

    except psycopg2.Error as e:
        logging.error(f"Database error: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

    return total_updated


# Example usage
if __name__ == "__main__":
    areas = [2, 3, 4]  # Replace with the ID of the search area you want to process
    for area_id in areas:
        updated_count = process_given_region(area_id)
    logging.info(f"Total number of search_boxes updated for area {area_id}: {updated_count}")
