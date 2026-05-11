import os
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Relative path to data directory (from repo root)
DATA_DIR = Path(__file__).parent.parent / "data"

def parse_filename(filename: str):
    """
    Parses the filename to extract metadata.
    Expected format: {id}_{date}_{dilution}_{pattern}.npy
    Example: 022541326800_20180810B1-2_1：320_peripheralcytoplasm.npy
    """
    # Remove extension
    name = filename.replace('.npy', '')
    
    parts = name.split('_')
    
    if len(parts) >= 4:
        patient_id = parts[0]
        date_info = parts[1]
        dilution = parts[2]
        # the rest is the pattern (in case pattern contains underscores, though it shouldn't usually)
        pattern = "_".join(parts[3:])
        return {
            "id": patient_id,
            "date": date_info,
            "dilution": dilution,
            "pattern": pattern,
            "filename": filename
        }
    else:
        # Fallback or edge case handling
        return {
            "id": "unknown",
            "date": "unknown",
            "dilution": "unknown",
            "pattern": name,
            "filename": filename
        }

def load_all_features(data_dir=DATA_DIR, max_samples=None):
    """
    Loads features and parses metadata for all .npy files in the directory.
    Returns:
        metadata_list: List of dicts with parsed metadata
        cls_embeddings: numpy array of shape [N, 768]
        patch_embeddings_paths: list of paths to the original .npy file or loaded patch features
                                (Loading all 6500 * 256 * 768 float32 might take ~5GB of RAM,
                                so we might want to lazy-load patches if memory is an issue)
    """
    files = list(data_dir.glob("*.npy"))
    if max_samples:
        files = files[:max_samples]
        
    metadata_list = []
    cls_embeddings = []
    
    print(f"Loading features from {len(files)} files...")
    
    for f in tqdm(files, desc="Parsing and Loading CLS tokens"):
        meta = parse_filename(f.name)
        
        # Load the numpy array
        # Shape is [257, 768] (1 CLS token + 256 patch tokens)
        try:
            feats = np.load(f)
            cls_token = feats[0]  # The first token is the CLS token
            # patch_tokens = feats[1:] 
            
            metadata_list.append(meta)
            cls_embeddings.append(cls_token)
        except Exception as e:
            print(f"Error loading {f.name}: {e}")
            
    return metadata_list, np.array(cls_embeddings), files

if __name__ == "__main__":
    # Test loading
    metadata, cls_embeds, files = load_all_features(max_samples=100)
    print(f"Loaded {len(metadata)} samples. CLS shape: {cls_embeds.shape}")
    print("Sample metadata:", metadata[0])
