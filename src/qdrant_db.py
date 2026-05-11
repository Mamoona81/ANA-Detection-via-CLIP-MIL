import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.data_loader import load_all_features, DATA_DIR

COLLECTION_NAME = "ana_biomedclip"
VECTOR_SIZE = 1024

def init_qdrant(client: QdrantClient):
    """
    Initializes Qdrant collection if it doesn't exist.
    """
    if not client.collection_exists(COLLECTION_NAME):
        print(f"Creating collection {COLLECTION_NAME}...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    else:
        print(f"Collection {COLLECTION_NAME} already exists.")

def ingest_data(client: QdrantClient):
    """
    Loads all data and ingests it into Qdrant collection.
    """
    metadata_list, cls_embeddings, files = load_all_features(DATA_DIR)
    
    points = []
    print("Preparing points for Qdrant...")
    for i, (meta, cls_emb, file_path) in enumerate(zip(metadata_list, cls_embeddings, files)):
        # Convert path to string for payload
        meta["file_path"] = str(file_path.absolute())
        
        points.append(
            PointStruct(
                id=i,  # Integer ID
                vector=cls_emb.tolist(),
                payload=meta
            )
        )
        
    print(f"Uploading {len(points)} points to Qdrant...")
    # Upload in batches
    batch_size = 500
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )
        print(f"Uploaded batch {i // batch_size + 1}/{(len(points) + batch_size - 1) // batch_size}")
        
    print("Ingestion complete.")

if __name__ == "__main__":
    # Create persistent local Qdrant instance
    db_path = str(DATA_DIR / "qdrant_storage")
    print(f"Connecting to Qdrant at {db_path}...")
    client = QdrantClient(path=db_path)
    
    init_qdrant(client)
    
    # Check if empty before ingesting
    count = client.count(collection_name=COLLECTION_NAME).count
    if count == 0:
        ingest_data(client)
    else:
        print(f"Collection {COLLECTION_NAME} already contains {count} points. Skipping ingestion.")
        
    # Test retrieval
    print("Testing retrieval logic...")
    test_vec = [0.0] * VECTOR_SIZE # Dummy
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=test_vec,
        limit=3
    ).points
    for hit in hits:
        print(hit.payload, "score:", hit.score)
