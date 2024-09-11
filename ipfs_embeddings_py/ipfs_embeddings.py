from . import ipfs_multiformats
import requests
import subprocess
import os
import json
class ipfs_embeddings_py:
    def __init__(self, resources, metedata):
        self.multiformats = ipfs_multiformats_py(resources, metedata)
        self.ipfs_only_hash = ipfs_only_hash_py(resources, metedata)
        self.https_endpoints = {}
        self.libp2p_endpoints = {}
        self.cid_queue = iter([])
        self.knn_queue = iter([])
        self.cid_index = {}
        self.knn_index = {}
        self.endpoint_status = {}
        self.add_https_endpoint = self.add_https_endpoint
        self.rm_https_endpoint = self.rm_https_endpoint
        self.queue_index_cid = self.queue_index_cid
        self.queue_index_knn = self.queue_index_knn
        self.choose_endpoint = self.choose_endpoint
        self.pop_index_knn = self.pop_index_knn
        self.pop_index_cid = self.pop_index_cid
        return None
    
    def load_index(self, index):
        self.index = index
        return None 
    
    def add_https_endpoint(self, model, endpoint, batch_size):
        if model not in self.https_endpoints:
            self.https_endpoints[model] = {}
            self.endpoint_status[endpoint] = 1
        if endpoint not in self.https_endpoints[model]:  
            self.https_endpoints[model][endpoint] = batch_size
            self.endpoint_status[endpoint] = 1
        return None
    
    def add_libp2p_endpoint(self, model, endpoint, batch_size):
        if model not in self.libp2p_endpoints:
            self.libp2p_endpoints[model] = {}
            self.endpoint_status[endpoint] = 1
        if endpoint not in self.libp2p_endpoints[model]:  
            self.libp2p_endpoints[model][endpoint] = batch_size
            self.endpoint_status[endpoint] = 1
        return None
    
    def rm_https_endpoint(self, model, endpoint):
        if model in self.https_endpoints and endpoint in self.https_endpoints[model]:
            del self.https_endpoints[model][endpoint]
            del self.endpoint_status[endpoint]
        return None
    
    def rm_libp2p_endpoint(self, model, endpoint):
        if model in self.libp2p_endpoints and endpoint in self.libp2p_endpoints[model]:
            del self.libp2p_endpoints[model][endpoint]
            del self.endpoint_status[endpoint]
        return None
    
    def test_tei_https_endpoint(self, model, endpoint):
        if model in self.https_endpoints and endpoint in self.https_endpoints[model]:
            return True
        return False

    def test_libp2p_endpoint(self, model, endpoint):
        if model in self.libp2p_endpoints and endpoint in self.libp2p_endpoints[model]:
            return True
        return False

    def get_https_endpoint(self, model):
        if model in self.https_endpoints:
            return self.https_endpoints[model]
        return None
    
    def get_libp2p_endpoint(self, model):
        if model in self.libp2p_endpoints:
            return self.libp2p_endpoints[model]
        return None

    def request_https_endpoint(self, model, batch_size):
        if model in self.https_endpoints:
            for endpoint in self.https_endpoints[model]:
                if self.endpoint_status[endpoint] == 1:
                    return endpoint
        return None

    def index_cid(self, samples):
        results = []
        if type(samples) is None:
            raise ValueError("samples must be a list")
        if type(samples) is str:
            samples = [samples]
        if type(samples) is iter:
            for this_sample in samples:
                this_sample_cid = self.multiformats.get_cid(this_sample)
                self.cid_index[this_sample_cid] = this_sample
            pass
        if type(samples) is list:
            for this_sample in samples:
                this_sample_cid = self.multiformats.get_cid(this_sample)
                self.cid_index[this_sample_cid] = this_sample
                results.append(this_sample_cid)
        return results
    
    def index_knn(self, samples, model):
        knn_stack = []
        if type(samples) is None:
            raise ValueError("samples must be a list")
        if type(samples) is str:
            samples = [samples]
        if type(samples) is iter:
            for this_sample in samples:
                chosen_endpoint = self.choose_endpoint(model)
                this_sample = {"inputs": i}
                query_response = self.make_post_request(chosen_endpoint, this_sample)
                knn_stack.append(query_response)
            pass
        if type(samples) is list:
            for this_sample in samples:
                chosen_endpoint = self.choose_endpoint(model)
                this_sample = {"inputs": i}
                query_response = self.make_post_request(chosen_endpoint, this_sample)
                knn_stack.append(query_response)
            pass
        return knn_stack
    
    def queue_index_cid(self, samples):
        if type(samples) is None:
            raise ValueError("samples must be a list")
        if isinstance(samples, str):
            samples = [samples]
            self.knn_queue = iter(list(samples))
            return True
        if isinstance(samples, list):
            self.knn_queue = list(self.knn_queue)
            for this_sample in samples:
                self.knn_queue.append(this_sample)
            self.knn_queue = iter(self.knn_queue)
            return True
        else:
            raise ValueError("samples must be a list")
    
    def queue_index_knn(self, samples):
        print("queue_index_knn")
        print(samples)
        print(type(samples))
        if samples is None:
            raise ValueError("samples must be a list")
        if isinstance(samples, str):
            samples = [samples]
            self.knn_queue = iter(list(samples))
            return True
        if isinstance(samples, list):
            self.knn_queue = list(self.knn_queue)
            for this_sample in samples:
                self.knn_queue.append(this_sample)
            self.knn_queue = iter(self.knn_queue)
            return True
        else:
            raise ValueError("samples must be a list")


    def make_post_request(endpoint, data):
        headers = {'Content-Type': 'application/json'}
        response = requests.post(endpoint, headers=headers, json=data)
        return response.json()

    def choose_endpoint(self, model):
        print("choose_endpoint")
        print(model)
        https_endpoints = self.get_https_endpoint(model)
        libp2p_endpoints = self.get_libp2p_endpoint(model)
        filtered_libp2p_endpoints = {k: v for k, v in self.endpoint_status.items() if v == 1}
        filtered_https_endpoints = {k: v for k, v in self.endpoint_status.items() if v == 1}
        if len(filtered_https_endpoints) == 0 and len(filtered_libp2p_endpoints) == 0:
            return None
        else:
            this_endpoint = None
            if len(list(filtered_https_endpoints.keys())) > 0:
                this_endpoint = list(filtered_https_endpoints.keys())[0]
            elif len(list(filtered_libp2p_endpoints.keys())) > 0:
                this_endpoint = list(filtered_libp2p_endpoints.keys())[0]
            return this_endpoint
        
    def https_index_cid(self, samples, endpoint):
        endpoint_chunk_size = self.https_endpoints[endpoint]
        all_chunk = []
        this_chunk = []
        for i in range(samples):
            self
            ## request endpoint
            pass
        return None
    
    def https_index_knn(self, selected_endpoint, model):
        batch_size = 0
        if "http" in selected_endpoint:
            batch_size = self.https_endpoints[model][selected_endpoint]
        elif "libp2p" in selected_endpoint:
            batch_size = self.libp2p_endpoints[model][selected_endpoint]
        knn_stack = []
        queue_knn = self.pop_index_knn(batch_size)
        json_queue_knn = json.dumps(queue_knn)
        for i in queue_knn:
            this_sample = {"inputs": i}
            query_response = self.make_post_request(selected_endpoint, this_sample)
            knn_stack.append(query_response)
        return knn_stack
    
    def select_endpoint(self, model):
        if model in self.https_endpoints:
            for endpoint in self.https_endpoints[model]:
                if self.endpoint_status[endpoint] == 1:
                    self.endpoint_status[endpoint] = 0
                    return endpoint
        return None

    def pop_index_cid(self, number):
        results = []
        if number > len(self.cid_queue):
            raise ValueError("number is greater than the queue size")
        if number <= 0:
            raise ValueError("number must be greater than 0")
        for i in range(number):
            results.push(self.cid_queue.pop())
            i += 1
        return results
    
    def pop_index_knn(self, number):
        results = []
        knn_queue_list = list(self.knn_queue)
        if number > len(knn_queue_list):
            raise ValueError("number is greater than the queue size")
        if number <= 0:
            raise ValueError("number must be greater than 0")
        for i in range(number):
            results.append(knn_queue_list.pop())
            i += 1
        self.knn_queue = iter(knn_queue_list)
        return results


    def test(self):
        self.https_endpoints("BAAI/bge-m3", "62.146.169.111:80/embed",1)
        self.https_endpoints("BAAI/bge-m3", "62.146.169.111:8080/embed",1)
        self.https_endpoints("BAAI/bge-m3", "62.146.168.111:8081/embed",1)
        test_knn_index = {}
        test_cid_index = {}
        test_data = {
            "test1", "test2", "test3"
        }

        for data in test_data:
            test_cid_index = self.index_ipfs(data)
            test_knn_index = self.index_knn(data)
        

        print("test")

    def status(self):
        return self.endpoint_status
    
    def setStatus(self,endpoint , status):
        self.endpoint_status[endpoint] = status
        return None

if __name__ == '__main__':
    resources = {}
    metedata = {}
    ipfs_embeddings = ipfs_embeddings_py(resources, metedata)
    ipfs_embeddings.test()
    print("test")