import os
import numpy as np
import json

# Paths
embeddings_path = "src/embeddings/embeddings.npz"
label_map_path = "src/embeddings/label_map.json"

# Remove old files
if os.path.exists(embeddings_path):
    os.remove(embeddings_path)
    print(f"Deleted: {embeddings_path}")

if os.path.exists(label_map_path):
    os.remove(label_map_path)
    print(f"Deleted: {label_map_path}")

# Create empty embeddings and label map
np.savez(embeddings_path, embeddings=np.empty((0, 128)))
with open(label_map_path, "w") as f:
    json.dump({}, f)

print("Embeddings and label map reset successfully.")
