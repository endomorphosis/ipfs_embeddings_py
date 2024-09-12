
from datasets import load_dataset
import ipfs_embeddings_py.ipfs_embeddings as ipfs_embeddings

class test_ipfs_embeddings:
    def __init__(self, resources, metadata):
        resources = {}
        metadata = {}
        self.dataset = {}
        self.ipfs_embeddings_py = ipfs_embeddings.ipfs_embeddings_py(resources, metadata)
        self.ipfs_embeddings_py.add_https_endpoint("BAAI/bge-m3", "http://62.146.169.111:80/embed",1)
        return None
    
    def test(self, model, endpoint):
        batch_size = self.ipfs_embeddings_py.max_batch_size(model, endpoint)
        return batch_size
    
if __name__ == '__main__':
    metadata = {
        "dataset": "laion/Wikipedia-X-Concat",
        "faiss_index": "laion/Wikipedia-M3",
        "model": "BAAI/bge-m3"
    }
    resources = {
        "https_endpoints": [["BAAI/bge-m3", "http://62.146.169.111:80/embed",1]]
    }
    test = test_ipfs_embeddings(resources, metadata)
    test.test(metadata["model"], resources["https_endpoints"][0][1])
    print("Test passed")
