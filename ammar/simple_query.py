################
##################
# Checking Query and testing our index
#########################
import pyarrow.parquet as pq
import numpy as np
import faiss

index = faiss.read_index('/mnt/nvme/ammar_faiss/Faiss_temporary')

file_path = '/mnt/nvme/ammar_faiss/dewiki_concat_embedding.parquet'
table = pq.read_table(file_path)

embeddings_column = table['Embeddings'].to_numpy()

query_embedding = np.array(embeddings_column[0])

if isinstance(query_embedding[0], list) or isinstance(query_embedding[0], np.ndarray):
    query_embedding = np.array(query_embedding[0])

query_embedding = query_embedding.reshape(1, -1).astype('float32')

k = 3
distances, indices = index.search(query_embedding, k)

print("Query embedding shape:", query_embedding.shape)
print(f"Nearest neighbors (FAISS indices): {indices}")
print(f"Distances to neighbors: {distances}")

for i, idx in enumerate(indices[0]):
    if distances[0][i] == 0:
        print(f"Exact match found at FAISS index {idx}")
    else:
        print(f"Matched with FAISS index {idx} at distance {distances[0][i]}")
