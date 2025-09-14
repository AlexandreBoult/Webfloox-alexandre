import random
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, Range, MatchValue

def trim_classes_in_collection(
    client: QdrantClient,
    collection_name: str,
    class_payload_key: str,
    max_points_per_class: int
):
    """
    Randomly deletes points from a Qdrant collection for any class that has
    more than a specified number of points.

    Args:
        client (QdrantClient): The initialized Qdrant client.
        collection_name (str): The name of the collection to modify.
        class_payload_key (str): The payload key that contains the class name.
        max_points_per_class (int): The target number of points for each class.
    """
    print(f"Checking collection '{collection_name}' for classes to trim...")

    # A dictionary to store point IDs, grouped by their class
    points_by_class = {}

    try:
        # Retrieve all points from the collection. For large collections, this might need
        # to be done in batches using scroll. This example assumes a manageable size.
        # We only need the payload and the id.
        all_points = []
        next_page_offset = None
        while True:
            scroll_result = client.scroll(
                collection_name=collection_name,
                limit=1000,  # Retrieve a manageable number of points per batch
                with_vectors=True,
                with_payload=True,
                offset=next_page_offset
            )
            points, next_page_offset = scroll_result
            all_points.extend(points)
            if next_page_offset is None:
                break

        # Group points by the specified class payload key
        for point in all_points:
            class_name = point.payload.get(class_payload_key)
            if class_name:
                if class_name not in points_by_class:
                    points_by_class[class_name] = []
                points_by_class[class_name].append(point.id)

        # Iterate through each class and trim if necessary
        for class_name, point_ids in points_by_class.items():
            num_points = len(point_ids)
            if num_points > max_points_per_class:
                num_to_delete = num_points - max_points_per_class
                print(f"Class '{class_name}' has {num_points} points, which is over the limit of {max_points_per_class}.")
                print(f"Will randomly delete {num_to_delete} points.")

                # Randomly select a subset of point IDs to delete
                ids_to_delete = random.sample(point_ids, num_to_delete)

                # Use the list of point IDs to delete the selected points
                client.delete(
                    collection_name=collection_name,
                    points_selector=ids_to_delete
                )
                print(f"Successfully deleted {len(ids_to_delete)} points from class '{class_name}'.")
            else:
                print(f"Class '{class_name}' has {num_points} points. No action needed.")

    except Exception as e:
        print(f"An error occurred: {e}")


# --- Main execution block ---
if __name__ == "__main__":
    # --- IMPORTANT: Replace these with your actual details ---
    QDRANT_HOST = "localhost" # or your Qdrant host URL
    QDRANT_PORT = 6333
    COLLECTION_NAME = "movie_critics"
    CLASS_PAYLOAD_KEY = "movie_id"
    TARGET_POINT_COUNT = 11

    # Initialize the Qdrant client
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Call the function to trim the collection
    trim_classes_in_collection(
        client=client,
        collection_name=COLLECTION_NAME,
        class_payload_key=CLASS_PAYLOAD_KEY,
        max_points_per_class=TARGET_POINT_COUNT
    )

    print("Script finished.")
