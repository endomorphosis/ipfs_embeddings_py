### Untested FastAPI for the query
import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

my_index = faiss.read_index("knn.index")

class QueryVector(BaseModel):
    vector: list
    k: int

@app.post("/search/")
async def search_faiss(query: QueryVector):
    query_vector = np.array([query.vector], dtype=np.float32)
    distances, indices = my_index.search(query_vector, query.k)
    results = [
        {"rank": i+1, "vector_number": int(indice), "distance": float(dist)}
        for i, (dist, indice) in enumerate(zip(distances[0], indices[0]))
    ]
    return {"query": query.vector, "results": results}


#### uvicorn app:app --host 0.0.0.0 --port 8000
## curl -X 'POST' \
#'http://127.0.0.1:8000/search/' \
# -H 'Content-Type: application/json' \
#  -d '{"vector": [0.1, 0.2, 0.3, ...], "k": 5}'
#######
