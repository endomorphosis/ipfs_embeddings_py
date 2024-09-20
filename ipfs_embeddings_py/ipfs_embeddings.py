from .ipfs_multiformats import *
from .ipfs_only_hash import *
import requests
import subprocess
import json
import random
import datasets
import asyncio
from aiohttp import ClientSession
from datasets import load_dataset
import datasets
import os
import sys
import subprocess
from transformers import AutoTokenizer

class ipfs_embeddings_py:
    def __init__(self, resources, metedata):
        self.multiformats = ipfs_multiformats_py(resources, metedata)
        self.ipfs_only_hash = ipfs_only_hash_py(resources, metedata)
        self.https_endpoints = {}
        self.libp2p_endpoints = {}
        self.datasets = datasets.Dataset
        self.index =  {}
        self.queues = {}
        self.cid_list = []
        self.cid_queue = iter([])
        self.knn_queue = iter([])
        self.cid_index = {}
        self.knn_index = {}
        self.join_column = None
        self.tokenizer = {}
        self.endpoint_status = {}
        self.new_dataset = {}
        self.send_batch = self.send_batch
        self.save_to_disk = self.save_to_disk
        self.producer = self.producer
        self.consumer = self.consumer
        self.async_generator = self.async_generator
        self.add_https_endpoint = self.add_https_endpoint
        self.rm_https_endpoint = self.rm_https_endpoint
        self.queue_index_cid = self.queue_index_cid
        self.queue_index_knn = self.queue_index_knn
        self.choose_endpoint = self.choose_endpoint
        self.pop_index_knn = self.pop_index_knn
        self.pop_index_cid = self.pop_index_cid
        self.max_batch_size = self.max_batch_size
        return None
    
    def load_index(self, index):
        self.index = index
        return None 
    
    def add_https_endpoint(self, model, endpoint, context_length):
        if model not in self.https_endpoints:
            self.https_endpoints[model] = {}
            self.endpoint_status[endpoint] = 1
        if endpoint not in self.https_endpoints[model]:  
            self.https_endpoints[model][endpoint] = context_length
            self.endpoint_status[endpoint] = 1
        return None
    
    def add_libp2p_endpoint(self, model, endpoint, context_length):
        if model not in self.libp2p_endpoints:
            self.libp2p_endpoints[model] = {}
            self.endpoint_status[endpoint] = 1
        if endpoint not in self.libp2p_endpoints[model]:  
            self.libp2p_endpoints[model][endpoint] = context_length
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
    
    async def max_batch_size(self, model, endpoint=None):
        embed_fail = False
        exponent = 1
        batch = []
        batch_size = 2**exponent
        while embed_fail == False:
            while len(batch) < batch_size:
                batch.append("Hello World")
            try:
                embeddings = await self.index_knn(batch, model, endpoint)
                if not isinstance(embeddings[0], list):
                    raise Exception("Embeddings not returned as list")
                embed_fail = False
                exponent += 1
                batch_size = 2**exponent
            except:
                embed_fail = True
                pass
        return 2**(exponent-1)
    
    def index_knn(self, samples, model, chosen_endpoint=None):
        knn_stack = []
        if type(samples) is None:
            raise ValueError("samples must be a list")
        if type(samples) is str:
            samples = [samples]
        if type(samples) is iter:
            if chosen_endpoint is None:
                chosen_endpoint = self.choose_endpoint(model)
            this_query = {"inputs": samples}
            try:
                query_response = self.make_post_request(chosen_endpoint, this_query)
            except:
                raise Exception("error: " + query_response["error"])
            if isinstance(query_response, dict) and "error" in query_response.keys():
                raise Exception("error: " + query_response["error"])
            else:
                knn_stack = query_response
            pass
        if type(samples) is list:
            if chosen_endpoint is None:
                chosen_endpoint = self.choose_endpoint(model)
            this_query = {"inputs": samples}
            try:
                query_response = self.make_post_request(chosen_endpoint, this_query)
            except Exception as e:
                raise Exception(e)
            if isinstance(query_response, dict) and "error" in query_response.keys():
                raise Exception("error: " + query_response["error"])
            else:
                knn_stack = query_response
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


    async def make_post_request(self, endpoint, data):
        headers = {'Content-Type': 'application/json'}
        async with ClientSession() as session:
            async with session.post(endpoint, headers=headers, json=data) as response:
                if response.status != 200:
                    return ValueError(response)
                return await response.json()

    def choose_endpoint(self, model):
        https_endpoints = self.get_https_endpoint(model)
        libp2p_endpoints = self.get_libp2p_endpoint(model)
        filtered_libp2p_endpoints = {k: v for k, v in self.endpoint_status.items() if v == 1 and libp2p_endpoints is not None and k in list(libp2p_endpoints.keys())}
        filtered_https_endpoints = {k: v for k, v in self.endpoint_status.items() if v == 1 and https_endpoints is not None and k in list(https_endpoints.keys())}
        if ( not filtered_https_endpoints and not filtered_libp2p_endpoints):
            return None
        else:
            this_endpoint = None
            if len(list(filtered_https_endpoints.keys())) > 0:
                this_endpoint = random.choice(list(filtered_https_endpoints.keys()))
            elif len(list(filtered_libp2p_endpoints.keys())) > 0:
                this_endpoint = random.choice(list(filtered_https_endpoints.keys()))
            print("chosen endpoint for " + model + " is " + this_endpoint)
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

    async def async_generator(self, iterable):
        for item in iterable:
            yield item

    async def consumer(self, queue, column, batch_size, model_name):
        batch = []
        if model_name not in self.index.keys():
            self.index[model_name] = datasets.Dataset.from_dict({"cid": [], "embedding": []})
        while True:
            item = await queue.get()  # Wait for item
            batch.append(item)
            if len(batch) >= batch_size:
                # Process batch
                results = await self.send_batch(batch, column, model_name)
                for i in range(len(results)):
                    self.index[model_name] = self.index[model_name].add_item({"cid": batch[i]["cid"], "embedding": results[i]})
                batch = []  # Clear batch after sending
                self.saved = False
            queue.task_done()
        return None

    async def producer(self, dataset_stream, column, queues):
        async for item in self.async_generator(dataset_stream):
            # Assuming `item` is a dictionary with required data
            column_names = item.keys()
            this_cid = self.index_cid(item[column])[0]
            if "cid" not in column_names:
                item["cid"] = self.index_cid(item[column])[0]
            # Check if cid is in index
            if this_cid in self.cid_list:
                pass
            else:
                self.cid_list.append(this_cid)
                self.new_dataset = self.new_dataset.add_item(item)    
                for queue in queues.values():
                    await queue.put(item)  # Non-blocking put
        return None

    async def send_batch(self, batch, column, model_name):
        print(f"Sending batch of size {len(batch)} to model {model_name}")
        endpoint = self.choose_endpoint(model_name)
        model_context_length = self.https_endpoints[model_name][endpoint]
        new_batch = []
        for item in batch:
            if model_name not in self.tokenizer.keys():
                self.tokenizer[model_name] = AutoTokenizer.from_pretrained(model_name)
            this_item_tokens = len(self.tokenizer[model_name].encode(item[column]))
            if this_item_tokens > model_context_length:
                encoded_item = self.tokenizer[model_name](item[column], return_tensors="pt")["input_ids"].tolist()[0]
                truncated_encoded_item = encoded_item[:model_context_length]
                unencode_item = self.tokenizer[model_name].decode(truncated_encoded_item)
                new_batch.append(unencode_item)
            else:
                new_batch.append(item[column])
        results = None
        try:
            results = await self.index_knn(new_batch, model_name, endpoint)
        except Exception as e:
            raise e
        if isinstance(results, ValueError):
            error = results.args[0]
            if error.status == 413:
                if error.reason == "Payload Too Large":
                    error_content = error.content._buffer[0].decode("utf-8")
                    error_content = json.loads(error_content)
                    if "error" in error_content.keys() and "error_type" in error_content.keys():
                        if "Validation" in error_content["error_type"] and "must have less than" in error_content["error"]:
                            expected = int(error_content["error"].split("must have less than ")[1].split(" tokens")[0])
                            given = int(error_content["error"].split("Given: ")[1])
                            difference = given - expected
                            self.https_endpoints[model_name][endpoint] = self.https_endpoints[model_name][endpoint] - difference
                            for item in new_batch:
                                index = new_batch.index(item)
                                item = item[:self.https_endpoints[model_name][endpoint]]
                                new_batch[index] = item
                            results = await self.index_knn(new_batch, model_name, endpoint)
                            return results
            raise Exception("error: " + error_content["error"])
        else:
            return results

    async def save_to_disk(self, dataset, dst_path, models):
        self.saved = False
        while True:
            await asyncio.sleep(600)
            if self.saved == False:
                self.new_dataset.to_parquet(dst_path+"/"+dataset.replace("/","---")+".parquet")   
                for model in models:
                    self.index[model].to_parquet(dst_path+"/"+model.replace("/","---")+".parquet")
                self.saved = True
        return None

    def status(self):
        return self.endpoint_status
    
    def setStatus(self,endpoint , status):
        self.endpoint_status[endpoint] = status
        return None
    