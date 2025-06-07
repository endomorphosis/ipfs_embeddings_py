##### Testing our theory on 10 rows

import pyarrow.parquet as pq
import numpy as np
import faiss

file_path = '/mnt/nvme/ammar_faiss/dewiki_concat_embedding.parquet'
table = pq.read_table(file_path)

embeddings_column = table['Embeddings'].to_numpy()

for i in range(10):
    embedding = np.array(embeddings_column[i])
    
    if isinstance(embedding[0], list) or isinstance(embedding[0], np.ndarray):
        embedding = np.array(embedding[0])
    
    print(f"Embedding {i}: Shape {embedding.shape}")
    print(embedding)

embeddings = []
for e in embeddings_column:
    embedding = np.array(e)
    
    if isinstance(embedding[0], list) or isinstance(embedding[0], np.ndarray):
        embedding = np.array(embedding[0])
    
    embeddings.append(embedding)

embeddings = np.vstack(embeddings).astype('float32')

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

faiss.write_index(index, '/mnt/nvme/ammar_faiss/Faiss_temporary')

print("FAISS index 'Faiss_temporary' created and saved successfully.")
