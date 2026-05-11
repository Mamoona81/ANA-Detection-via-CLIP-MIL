import os
import random
import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient
from src.qdrant_db import COLLECTION_NAME, DATA_DIR
from src.data_loader import parse_filename

def evaluate_retrieval(client: QdrantClient, samples=100, top_k=5):
    """
    Evaluates Retrieval Recall@K on a random subset of cases.
    For each query, we check if the correct ANA pattern is within the Top-K retrieved results.
    """
    files = list(DATA_DIR.glob("*.npy"))
    random.seed(42)
    test_files = random.sample(files, min(samples, len(files)))
    
    correct_at_1 = 0
    correct_at_k = 0
    total = 0
    
    print(f"Evaluating retrieval on {len(test_files)} random samples...")
    for f in tqdm(test_files):
        meta = parse_filename(f.name)
        true_pattern = meta['pattern']
        
        try:
            feats = np.load(f)
            query_cls = feats[0]
            
            # Retrieve from Qdrant
            res = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_cls.tolist(),
                limit=top_k + 1 # +1 to exclude the exact match if it returns itself
            )
            
            # Filter out the self-match if it's there
            hits = [h for h in res.points if h.payload['file_path'] != str(f.absolute())]
            hits = hits[:top_k]
            
            if not hits:
                continue
                
            top1_pattern = hits[0].payload.get('pattern')
            topk_patterns = [h.payload.get('pattern') for h in hits]
            
            if top1_pattern == true_pattern:
                correct_at_1 += 1
            if true_pattern in topk_patterns:
                correct_at_k += 1
            
            total += 1
            
        except Exception as e:
            pass

    if total > 0:
        print(f"\n--- Retrieval Evaluation Results ({total} queries) ---")
        print(f"Recall@1: {correct_at_1 / total:.4f}")
        print(f"Recall@{top_k}: {correct_at_k / total:.4f}")
    else:
        print("No valid samples for evaluation.")

if __name__ == "__main__":
    db_path = str(DATA_DIR / "qdrant_storage")
    client = QdrantClient(path=db_path)
    try:
        evaluate_retrieval(client, samples=500, top_k=5)
    finally:
        client.close()
