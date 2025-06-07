import pyarrow.parquet as pq
import numpy as np

file_path = '/mnt/nvme/ammar_faiss/dewiki_concat_embedding.parquet'
table = pq.read_table(file_path)

embeddings_column = table['Embeddings'].to_numpy()

# Unpack the first embedding and check the shape
first_embedding = np.array(embeddings_column[0])
if first_embedding.ndim == 1 and len(first_embedding) == 1:
    first_embedding = np.array(first_embedding[0])

print(f"Shape of the embeddings: {first_embedding.shape}")
