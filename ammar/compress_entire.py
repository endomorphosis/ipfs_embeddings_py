# Compress and creating index for entire 2.5M rows

import pyarrow.parquet as pq
import numpy as np
import faiss

file_path = '/mnt/nvme/ammar_faiss/dewiki_concat_embedding.parquet'
table = pq.read_table(file_path)

embeddings_column = table['Embeddings'].to_numpy()

embeddings = []
for e in embeddings_column:
    embedding = np.array(e)
    if isinstance(embedding[0], list) or isinstance(embedding[0], np.ndarray):
        embedding = np.array(embedding[0])
    embeddings.append(embedding)

embeddings = np.vstack(embeddings).astype('float32')

dimension = embeddings.shape[1]
nlist = 100
m = 8
nsubquantizers = 8

quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nsubquantizers)

index.train(embeddings)
index.add(embeddings)

num_rows = embeddings.shape[0]
num_rows_millions = num_rows / 1000000

print(f"Number of rows added to FAISS index: {num_rows_millions:.2f} million")

faiss.write_index(index, '/mnt/nvme/ammar_faiss/Faiss_temporary')

print("FAISS index 'Faiss_temporary' created and saved successfully.")
