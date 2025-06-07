# Unpack and creating .npy file

import pyarrow.parquet as pq
import numpy as np
import os

file_path = '/mnt/nvme/ammar_faiss/dewiki_concat_embedding.parquet'
table = pq.read_table(file_path)

embeddings_column = table['Embeddings'].to_numpy()

# Unpack embeddings and save them
embeddings = []
for e in embeddings_column:
    embedding = np.array(e)
    if isinstance(embedding[0], list) or isinstance(embedding[0], np.ndarray):
        embedding = np.array(embedding[0])
    embeddings.append(embedding)

embeddings = np.vstack(embeddings).astype('float32')

# Save embeddings to a file
np.save('/mnt/nvme/ammar_faiss/german_embeddings.npy', embeddings)
