import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from qdrant_client import QdrantClient
from src.qdrant_db import COLLECTION_NAME, DATA_DIR

def normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(norm > 0, v / norm, v)

def get_retrieved_context(client: QdrantClient, query_cls: np.ndarray, top_k=3):
    """
    Retrieves top-k similar images from Qdrant.
    """
    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_cls.tolist(),
        limit=top_k
    )
    return res.points

def compute_patch_similarity_heatmap(query_patches: np.ndarray, ref_patches: np.ndarray):
    """
    Computes patch-to-patch similarity between a query image and a reference image.
    Both should have shape [256, 1024].
    Returns a 16x16 heatmap indicating the maximum similarity of each query patch to the reference.
    """
    # Normalize
    q_norm = normalize(query_patches)
    r_norm = normalize(ref_patches)
    
    # Cosine similarity matrix: [256, 256]
    sim_matrix = np.dot(q_norm, r_norm.T)
    
    # For each patch in query, find the max similarity to any patch in the reference
    # This shows which parts of the query matched best with the reference
    max_sim_per_patch = np.max(sim_matrix, axis=1) # Shape: [256]
    
    # Reshape to 16x16
    heatmap = max_sim_per_patch.reshape(16, 16)
    return heatmap

def save_heatmap_image(heatmap: np.ndarray, output_path: str, title="Patch Similarity"):
    """
    Saves the 16x16 heatmap as an image for Gemini.
    """
    plt.figure(figsize=(4, 4))
    plt.imshow(heatmap, cmap='viridis', interpolation='nearest')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def build_rag_prompt_and_visuals(query_file: str, client: QdrantClient, output_dir: str, top_k=3):
    """
    Retrieves context and builds the multimodal prompt components.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Query
    try:
        query_feats = np.load(query_file)
    except Exception as e:
        print(f"Error loading {query_file}: {e}")
        return None, []
        
    query_cls = query_feats[0]
    query_patches = query_feats[1:] # Should be 256
    
    # 2. Retrieve
    hits = get_retrieved_context(client, query_cls, top_k=top_k)
    
    # 3. Build Prompt & Visuals
    prompt = "You are an expert AI rheumatologist analyzing a patient's HEp-2 immunofluorescence image.\n"
    prompt += "Due to privacy constraints, we cannot view the raw image. Instead, we present you with patch-wise similarity heatmaps derived from BiomedCLIP ViT-L/14 patch features.\n\n"
    prompt += "We performed a vector search to find similar historical patient cases. Here is the context:\n\n"
    
    heatmap_paths = []
    
    for i, hit in enumerate(hits):
        ref_meta = hit.payload
        ref_file = ref_meta['file_path']
        score = hit.score
        
        prompt += f"--- Reference Case {i+1} ---\n"
        prompt += f"Similarity Score: {score:.4f}\n"
        prompt += f"ANA Pattern: {ref_meta.get('pattern', 'Unknown')}\n"
        prompt += f"Dilution: {ref_meta.get('dilution', 'Unknown')}\n"
        prompt += f"Patient ID: {ref_meta.get('id', 'Unknown')}\n\n"
        
        # Load ref patches for explainability
        if os.path.exists(ref_file):
            ref_feats = np.load(ref_file)
            ref_patches = ref_feats[1:]
            
            # Generate heatmap
            heatmap = compute_patch_similarity_heatmap(query_patches, ref_patches)
            hm_path = os.path.join(output_dir, f"heatmap_ref_{i+1}.png")
            save_heatmap_image(heatmap, hm_path, title=f"Match vs Case {i+1}\n({ref_meta.get('pattern')})")
            heatmap_paths.append(hm_path)
        else:
            print(f"Warning: Ref file {ref_file} not found.")
            
    prompt += "The attached images are 16x16 heatmaps showing which regions of the query image matched mostly strongly with each reference case. Yellow/Brighter regions indicate higher similarity.\n\n"
    prompt += "Based on these most similar cases and their patterns, predict the most likely ANA pattern for the primary query image, and explain your clinical diagnostic reasoning using the similarity scores and metadata."
    
    return prompt, heatmap_paths

if __name__ == "__main__":
    db_path = str(DATA_DIR / "qdrant_storage")
    client = QdrantClient(path=db_path)
    try:
        # Grab a random file for testing
        files = list(DATA_DIR.glob("*.npy"))
        if files:
            test_file = str(files[0])
            print(f"Testing RAG pipeline with query: {test_file}")
            prompt, paths = build_rag_prompt_and_visuals(test_file, client, output_dir=str(DATA_DIR / "output"))
            
            print("\n--- GENERATED PROMPT ---")
            print(prompt)
            print("Generated heatmap paths:", paths)
    finally:
        client.close()
