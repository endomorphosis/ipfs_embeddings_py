import pyarrow.parquet as pq
import numpy as np

file_path = '/mnt/nvme/ammar_faiss/dewiki_concat_embedding.parquet'
table = pq.read_table(file_path)

embeddings_column = table['Embeddings'].to_numpy()
embeddings_shape = np.array(embeddings_column[0]).shape
print(f"Shape of the embeddings: {embeddings_shape}")
