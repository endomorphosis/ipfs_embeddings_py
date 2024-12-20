import os
import sys
import json
import random
import datasets
import asyncio
import subprocess
import aiohttp
import requests
import torch
import faiss
import numpy as np
from aiohttp import ClientSession, ClientTimeout
from multiprocessing import Pool
from transformers import AutoTokenizer
import datasets
from transformers import AutoModel
from datasets import Dataset, concatenate_datasets, load_dataset
import ipfs_multiformats
try:
    from . import chunker
except:
    from .chunker import chunker
import time
from ipfs_accelerate_py import ipfs_accelerate_py
from .ipfs_multiformats import ipfs_multiformats_py

class ipfs_embeddings_py:
    def __init__(self, resources, metadata):
        self.multiformats = ipfs_multiformats_py(resources, metadata)
        self.datasets = datasets.Dataset
        self.chunker = chunker.chunker(resources, metadata)
        # self.elasticsearch = elasticsearch_kit(resources, metadata)
        self.consumer_task_done = {}
        self.producer_task_done = False
        self.save_to_disk_task_done = False
        self.https_endpoints = {}
        self.libp2p_endpoints = {}
        self.index =  {}
        self.queues = {}
        self.caches = {}
        self.chunk_cache = {}
        self.chunk_embeddings = {}
        self.cid_chunk_list = []
        self.cid_chunk_set = set()
        self.batch_sizes = {}
        self.cid_list = set()
        self.cid_set = set()
        self.new_dataset = None
        self.all_cid_list = {}
        self.all_cid_set = {}
        self.cid_chunk_queue = None
        self.cid_index = {}
        self.knn_index = {}
        self.join_column = None
        self.tokenizer = {}
        self.endpoint_status = {}
        self.new_dataset = {}
        self.new_dataset_children = {}
        self.saved = False
        self.resources = resources
        self.metadata = metadata
        self.index_dataset = self.index_dataset
        self.index_knn = self.index_knn
        self.make_post_request = self.make_post_request
        self.choose_endpoint = self.choose_endpoint
        self.get_endpoints = self.get_endpoints
        self.max_batch_size = self.max_batch_size
        self.consumer = self.consumer
        self.producer = self.producer
        self.process_item = self.process_item
        self.save_checkpoints_to_disk = self.save_checkpoints_to_disk
        self.status = self.status
        self.setStatus = self.setStatus
        self.index_cid = self.index_cid
        self.load_index = self.load_index
        self.async_generator = self.async_generator
        self.send_batch_to_endpoint = self.send_batch_to_endpoint
        # Initialize endpoints
        return None

    def load_index(self, index):
        self.index = index
        return None 
    
    async def init(self):
        await self.ipfs_accelerate.init()
        
    

    def index_cid(self, samples):
        results = []
        if samples is None:
            raise ValueError("samples must be a list")
        if isinstance(samples, str):
            samples = [samples]
        if isinstance(samples, list):
            for this_sample in samples:
                this_sample_cid = self.multiformats.get_cid(this_sample)
                self.cid_index[this_sample_cid] = this_sample
                results.append(this_sample_cid)
        else:
            raise ValueError("samples must be a list or string")
        return results

    async def index_knn(self, samples, model, chosen_endpoint=None):
        knn_stack = []
        if chosen_endpoint is None:
            chosen_endpoint = self.choose_endpoint(model)
        if type(samples) is None:
            raise ValueError("samples must be a list")
        if type(samples) is str:
            samples = [samples]
        if type(samples) is list or type(samples) is iter:
            this_query = {"inputs": samples}
            if chosen_endpoint is None:
                if "chosen_endpoint" not in list(dir(self)) or self.chosen_local_endpoint is None or self.chosen_local_endpoint_model != model:
                    self.chosen_local_endpoint_model = model
                    self.chosen_local_endpoint = AutoModel.from_pretrained(model)
                    if model not in self.tokenizer.keys():
                        self.tokenizer[model] = AutoTokenizer.from_pretrained(model, device='cpu', use_fast=True)
                chosen_endpoint = self.chosen_local_endpoint
                chosen_endpoint.eval()
                inputs = self.tokenizer[model](samples, return_tensors="pt")
                with torch.no_grad():
                    query_response = chosen_endpoint(**inputs).last_hidden_state
                    query_response = query_response.tolist()[0]
            else:
                try:
                    query_response = await self.make_post_request(chosen_endpoint, this_query)
                except Exception as e:
                    print(str(e))
                    if "413" in str(e):
                        return ValueError(e)
                    if "can not write request body" in str(e):
                        return ValueError(e)
                    return ValueError(e)
            
            if isinstance(query_response, dict) and "error" in query_response.keys():
                raise Exception("error: " + query_response["error"])
            else:
                knn_stack = query_response
            pass
        return knn_stack

    async def make_post_request(self, endpoint, data):
        headers = {'Content-Type': 'application/json'}
        timeout = ClientTimeout(total=300) 
        async with ClientSession(timeout=timeout) as session:
            try:
                async with session.post(endpoint, headers=headers, json=data) as response:
                    if response.status != 200:
                        return ValueError(response)
                    return await response.json()
            except Exception as e:
                print(str(e))
                if "Can not write request body" in str(e):
                    print( "endpoint " + endpoint + " is not accepting requests")
                    return ValueError(e)
                if "Timeout" in str(e):
                    print("Timeout error")
                    return ValueError(e)
                if "Payload is not completed" in str(e):
                    print("Payload is not completed")
                    return ValueError(e)
                if "Can not write request body" in str(e):
                    return ValueError(e)
                pass
            except aiohttp.ClientPayloadError as e:
                print(f"ClientPayloadError: {str(e)}")
                return ValueError(f"ClientPayloadError: {str(e)}")
            except asyncio.TimeoutError as e:
                print(f"Timeout error: {str(e)}")
                return ValueError(f"Timeout error: {str(e)}")
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                return ValueError(f"Unexpected error: {str(e)}")
        

    async def async_generator(self, iterable):
        for item in iterable:
            yield item

    async def consumer(self, queue, column, batch_size, model_name, endpoint):
        print("consumer started for model " + model_name + " at endpoint " + endpoint)
        self.consumer_task_done[(model_name, endpoint)] = False
        batch = []
        if model_name not in self.caches.keys():
            self.caches[model_name] = {"items" : []}
        if model_name not in self.index.keys():
            self.index[model_name] = datasets.Dataset.from_dict({"cid": [], "embedding": []})
        while True:
            item = await queue.get()  # Wait for item
            batch.append(item)
            if len(batch) >= batch_size:
                # Process batch
                results = await self.send_batch_to_endpoint(batch, column, model_name, endpoint)
                for i in range(len(results)):
                    self.caches[model_name]["items"].append({"cid": batch[i]["cid"], "embedding": results[i]})
                batch = []  # Clear batch after sending
                self.saved = False
            queue.task_done()
            if self.producer_task_done and queue.empty():
                self.consumer_task_done[(model_name, endpoint)] = True
                break
        return None

    async def chunk_producer(self, dataset_stream, column, method=None, tokenizer=None, chunk_size=None, n_sentences=None, step_size=None, embed_model=None):
        chunk_tasks = []
        async for item in self.async_generator(dataset_stream):
            chunked_item = await self.chunk_item(item, column, method, tokenizer, chunk_size, n_sentences, step_size, embed_model)
            if chunked_item["parent_cid"] not in self.cid_chunk_set:
                while self.cid_chunk_queue.full():
                    await asyncio.sleep(0.1)
                if not self.cid_chunk_queue.full():
                    self.cid_chunk_queue.put_nowait(chunked_item)
                    pass
        return None

    async def chunk_consumer(self, batch_size, model_name, endpoint):
        print("chunk consumer started")
        while True:
            chunked_item = await self.cid_chunk_queue.get()
            batch_results = []
            batch = []
            chunk_data = []
            if chunked_item is not None:
                for item in chunked_item["items"]:
                    batch.append(item)
                    chunk_data.append(item)
                    if len(batch) >= batch_size or len(batch) == len(chunked_item["items"]):
                        results = await self.send_batch_to_endpoint(batch, "content", model_name, endpoint)
                        for i in range(len(results)):
                            batch_results.append({"cid": batch[i]["cid"], "index": chunk_data[i]["index"], "content": chunk_data[i]["content"] , "embedding": results[i]})
                        batch = []
                        chunk_data = []
            if len(batch_results) > 0:
                self.chunk_cache[chunked_item["parent_cid"]] = {"items": batch_results, "parent_cid": chunked_item["parent_cid"]}
                self.cid_chunk_set.add(chunked_item["parent_cid"])
                self.cid_chunk_list.append(chunked_item["parent_cid"])
                self.cid_chunk_queue.task_done()
                self.saved = False

    async def producer(self, dataset_stream, column, queues):
        tasks = []
        self.producer_task_done = False
        async for item in self.async_generator(dataset_stream):
            task = self.process_item(item, column, queues)
            tasks.append(task)
            if len(tasks) >= 1:
                await asyncio.gather(*tasks)
                tasks = []
        if tasks:
            await asyncio.gather(*tasks)
        self.producer_task_done = True
        return None
    
    async def chunk_item(self, item, column=None, method=None, tokenizer=None, chunk_size=None, n_sentences=None, step_size=None, embed_model=None):
        # Assuming `item` is a dictionary with required data
        if column is None:
            content = json.dumps(item)
        elif column not in list(item.keys()):
            content = json.dumps(item)
        else:
            content = item[column]
        if embed_model is None:
            if len(self.metadata["models"]) == 0:
                embed_model = "thenlper/gte-small"
            else:
                embed_model = self.metadata["models"][0]
        if chunk_size is None:
            chunk_size = 512
        if n_sentences is None:
            n_sentences = 8
        if step_size is None:
            step_size = 256
        if tokenizer is None:
            if len(list(self.tokenizer.keys())) == 0:
                self.tokenizer = AutoTokenizer.from_pretrained(embed_model, device='cpu', use_fast=True)
            else:
                tokenizer = self.tokenizer[list(self.tokenizer.keys())[0]]
        if method is None:
            fixed_chunk_list = self.chunker.chunk(content, self.tokenizer[list(self.tokenizer.keys())[0]], "fixed", 512, 8, 256, self.metadata["models"][0]) 
            semantic_chunk_list = self.chunker.chunk(content, self.tokenizer[list(self.tokenizer.keys())[0]], "semantic", 512, 8, 256, self.metadata["models"][0])
            sentences_chunk_list = self.chunker.chunk(content, self.tokenizer[list(self.tokenizer.keys())[0]], "sentences", 512, 8, 256, self.metadata["models"][0] )
            sliding_window_chunk_list = self.chunker.chunk(content, self.tokenizer[list(self.tokenizer.keys())[0]], "sliding_window", 512, 8, 256, self.metadata["models"][0])
            content_chunks = fixed_chunk_list + semantic_chunk_list + sentences_chunk_list + sliding_window_chunk_list
        else:
            content_chunks = self.chunker.chunk(content, tokenizer, method, chunk_size, n_sentences, step_size, embed_model)
        parent_cid = item["items"]["cid"]
        content_tokens = tokenizer.encode(content)
        ## sort content_chunks by the firt element of each tuple then the second element
        content_chunks = sorted(content_chunks, key=lambda x: (x[0], x[1]))
        ## filter out chunks that are larger than the chunk_size
        content_chunks = [chunk for chunk in content_chunks if chunk[1] - chunk[0] <= chunk_size]
        ## filter content_chunks to remove duplicates
        seen_chunks = set()
        unique_content_chunks = []
        for chunk in content_chunks:
            if chunk not in seen_chunks:
                unique_content_chunks.append(chunk)
                seen_chunks.add(chunk)
        content_chunks = unique_content_chunks
        if parent_cid in list(self.caches.keys()):
            pass
        else:
            cid_chunks = {"items" : [], "parent_cid": parent_cid}
            for chunk in content_chunks:
                chunk_index = chunk
                chunk_content = content_tokens[chunk[0]:chunk[1]]
                chunk_text = tokenizer.decode(chunk_content)
                child_cid = self.multiformats.get_cid(chunk_text)
                child_content = {"cid": child_cid, "index": chunk_index, "content": chunk_text}
                cid_chunks["items"].append(child_content)
        return cid_chunks
        
    async def process_item(self, item, column=None, queues=None):
        # Assuming `item` is a dictionary with required data
        if "new_dataset" not in list(self.caches.keys()):
            self.caches["new_dataset"] = {"items" : []}
        # print(f"Processing item with CID {index_cid(item[column])[0]}")
        if queues is None:
            queues = self.queues
        column_names = item.keys()
        if column is None:
            this_cid = self.index_cid(json.dumps(item))[0]
        elif column not in column_names:
            this_cid = self.index_cid(json.dumps(item))[0]
        else:
            this_cid = self.index_cid(item[column])[0]
        if "cid" not in column_names:
            item["cid"] = this_cid
        elif item["cid"] is None:
            item["cid"] = this_cid
        # Check if cid is in index
        if this_cid in self.cid_set:
            # print(f"CID {this_cid} already in index, skipping item.")
            return None
        else:
            self.cid_set.add(this_cid)
            if this_cid not in self.all_cid_set["new_dataset"]:
                self.caches["new_dataset"]["items"].append(item)
                self.saved = False
            models = self.queues.keys()
            for model, model_queues in queues.items():
                if len(model_queues) > 0:
                    if this_cid not in self.all_cid_set[model]:
                        endpoint, queue = min(model_queues.items(), key=lambda x: x[1].qsize())
                        while queue.full():
                            await asyncio.sleep(0.1)
                        queue.put_nowait(item)  # Non-blocking put
            return item

    async def send_batch_to_endpoint(self, batch, column, model_name, endpoint):
        print(f"Sending batch of size {len(batch)} to model {model_name} at endpoint {endpoint}")
        model_context_length = self.https_endpoints[model_name][endpoint]
        new_batch = []
        if model_name not in self.tokenizer.keys():
            self.tokenizer[model_name] = AutoTokenizer.from_pretrained(model_name, device='cpu')
        for item in batch:
            if column in list(item.keys()):
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
            print(e)
            pass
            # raise e
        if isinstance(results, ValueError):
            error = results.args[0]
            strerror = None
            if "strerror" in dir(error):
                strerror = error.strerror
            if "status" in dir(error):
                if error.status == 413:
                    if error.reason == "Payload Too Large":
                        error_content = error.content._buffer[0].decode("utf-8")
                        error_content = json.loads(error_content)
                        if "error" in error_content.keys() and "error_type" in error_content.keys():
                            if "Validation" in error_content["error_type"] and "must have less than" in error_content["error"]:
                                expected = int(error_content["error"].split("must have less than ")[1].split(" tokens")[0])
                                given = int(error_content["error"].split("Given: ")[1])
                                difference = given - expected
                                self.https_endpoints[model_name][endpoint] = model_context_length - difference
                                for item in new_batch:
                                    index = new_batch.index(item)
                                    item = { column : item[:self.https_endpoints[model_name][endpoint]] }
                                    new_batch[index] = item
                                results = await self.send_batch_to_endpoint(new_batch, column, model_name, endpoint)
                                return results
                            if "Validation" in error_content["error_type"] and "cannot be empty":
                                print("error: " + error_content["error"])
                                return None
                elif error.status == 504 or error.status == 502 or  "can not write request body" in str(error):
                    # self.endpoint_status[endpoint] = 0
                    new_endpoint = self.choose_endpoint(model_name)
                    if new_endpoint:
                        # new_queue = self.queues[model_name][new_endpoint]
                        # for item in batch:
                        #     await new_queue.put(item)
                        return await self.send_batch_to_endpoint(batch, column, model_name, new_endpoint)
                    else:
                        return await self.send_batch_to_endpoint(batch, column, model_name, endpoint)
                elif error.status == 400 or error.status == 404:
                    return await self.send_batch_to_endpoint(batch, column, model_name, endpoint)
            elif "Can not write request body" in error.strerror or "Timeout" in error.strerror:
                # self.endpoint_status[endpoint] = 0
                new_endpoint = self.choose_endpoint(model_name)
                if new_endpoint:
                    # new_queue = self.queues[model_name][new_endpoint]
                    # for item in batch:
                    #     await new_queue.put(item)
                    return await self.send_batch_to_endpoint(batch, column, model_name, new_endpoint)
                else:
                    return await self.send_batch_to_endpoint(batch, column, model_name, endpoint)
            raise Exception(error) 
        else:
            if results is None:
                return await self.send_batch_to_endpoint(batch, column, model_name, endpoint)
            print(f"Received embeddings for {len(results)} items from model {model_name} at endpoint {endpoint}")
            return results

    async def save_checkpoints_to_disk(self, dataset, dst_path, models):
        self.saved = False
        while True:
            await asyncio.sleep(60)
            if self.saved == False:
                if not os.path.exists(os.path.join(dst_path, "checkpoints")):
                    os.makedirs(os.path.join(dst_path, "checkpoints"))
                if not os.path.exists(os.path.join(dst_path, "checkpoints", "sparse_chunks")):
                    os.makedirs(os.path.join(dst_path, "checkpoints", "sparse_chunks"))
                if not os.path.exists(os.path.join(dst_path, "checkpoints", "sparse_embeddings")):
                    os.makedirs(os.path.join(dst_path, "checkpoints", "sparse_embeddings"))
                ls_checkpoints = os.listdir(os.path.join(dst_path, "checkpoints"))
                if self.caches["new_dataset"] and len(self.caches["new_dataset"]["items"]) > 0:
                    tmp_dataset = datasets.Dataset.from_dict(self.caches["new_dataset"])
                    tmp_dataset_cids = tmp_dataset.map(lambda x: {"cid": x["items"]["cid"]})["cid"]
                    self.all_cid_list["new_dataset"] += tmp_dataset_cids
                    self.all_cid_set["new_dataset"] = set(self.all_cid_set["new_dataset"].union(set(tmp_dataset_cids)))
                    tmp_dataset_cids_dataset = datasets.Dataset.from_dict({"cids": tmp_dataset_cids})
                    new_dataset_shards = [x for x in ls_checkpoints if "ipfs_" + dataset.replace("/", "___") + "_shard" in x and "_cids" not in x]
                    next_filename_shard = f"ipfs_{dataset.replace('/', '___')}_shard_{len(new_dataset_shards)}"
                    tmp_dataset_cids_dataset.to_parquet(os.path.join(dst_path, "checkpoints", next_filename_shard + "_cids.parquet"))
                    tmp_dataset.to_parquet(os.path.join(dst_path, "checkpoints", next_filename_shard + ".parquet"))
                    del tmp_dataset
                    del tmp_dataset_cids
                    del tmp_dataset_cids_dataset
                    del self.caches["new_dataset"]
                    self.caches["new_dataset"] = {"items" : []}
                for model in models:
                    if model in self.caches.keys():
                        if self.caches[model] and len(self.caches[model]["items"]) > 0:
                            tmp_dataset = datasets.Dataset.from_dict(self.caches[model])
                            tmp_dataset_cids = tmp_dataset.map(lambda x: {"cid": x["items"]["cid"]})["cid"]
                            self.all_cid_list[model] += tmp_dataset_cids
                            self.all_cid_set[model] = set(self.all_cid_set[model].union(set(tmp_dataset_cids)))
                            tmp_dataset_cids_dataset = datasets.Dataset.from_dict({"cids": list(tmp_dataset_cids)})
                            self.caches[model] = {"items" : []}
                            this_model_shards = [x for x in ls_checkpoints if model.replace("/", "___") + "_shard" in x and "_cids" not in x]
                            next_filename_shard = f"{dataset.replace('/', '___')}_{model.replace('/', '___')}_shard_{len(this_model_shards)}"
                            tmp_dataset.to_parquet(os.path.join(dst_path, "checkpoints", next_filename_shard + ".parquet"))
                            tmp_dataset_cids_dataset.to_parquet(os.path.join(dst_path, "checkpoints", next_filename_shard + "_cids.parquet"))
                            print("Saved "+ str(len(tmp_dataset)) + " items to disk for model " + model + " at " + dst_path)
                            del tmp_dataset
                            del tmp_dataset_cids
                            del tmp_dataset_cids_dataset
                            self.caches[model] = {"items" : []}
                for this_cid in list(self.chunk_cache.keys()):
                    this_chunk = self.chunk_cache[this_cid]
                    this_cid_dataset = datasets.Dataset.from_dict({"items":this_chunk["items"]})
                    this_cid_dataset.to_parquet(os.path.join(dst_path, "checkpoints", "sparse_chunks", this_cid + ".parquet"))
                    print("Saved " + str(len(this_cid_dataset)) + " chunks to disk for CID " + this_cid + " at " + dst_path)
                    self.cid_chunk_set.add(this_cid)
                    self.cid_chunk_list.append(this_cid)
                    del self.chunk_cache[this_cid]
                    del this_cid_dataset
                self.saved = True
            # if self.producer_task_done and all(self.consumer_task_done.values()):
            #     self.save_to_disk_task_done = True
            #     break
        return None 

    def status(self):
        return self.endpoint_status

    def setStatus(self, endpoint, status):
        self.endpoint_status[endpoint] = status
        return None
    

    async def index_dataset(self, dataset, split, column, dst_path, models = None):
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        self.queues = {}
        self.cid_set = set()
        self.all_cid_list = {}
        consumer_tasks = {}
        batch_sizes = {}
        if models is None:
            models = list(self.https_endpoints.keys())
        for model in models:
            if model not in self.queues:
                self.queues[model] = {}
        if split is None:
            self.dataset = load_dataset(dataset, streaming=True).shuffle(random.randint(0,65536))
        else:
            self.dataset = load_dataset(dataset, split=split, streaming=True).shuffle(random.randint(0,65536))
        columns = self.dataset.column_names
        columns.append("cid")
        await self.load_checkpoints( dataset, split, dst_path, models)
        consumer_tasks = {}
        for model in models:
            endpoints = self.get_endpoints(model)
            if not endpoints:
                continue
            for endpoint in endpoints:
                batch_size = 0
                if model not in self.batch_sizes:
                    self.batch_sizes[model] = {}
                if model not in self.queues:
                    self.queues[model] = {}
                if endpoint not in list(self.batch_sizes[model].keys()):
                    batch_size = await self.max_batch_size(model, endpoint)
                    self.batch_sizes[model][endpoint] = batch_size
                if self.batch_sizes[model][endpoint] > 0:
                    self.queues[model][endpoint] = asyncio.Queue()  # Unbounded queue
                    consumer_tasks[(model, endpoint)] = asyncio.create_task(self.consumer(self.queues[model][endpoint], column, batch_size, model, endpoint))
        # Compute commonn
        self.cid_set = set.intersection(*self.all_cid_set.values())
        producer_task = asyncio.create_task(self.producer(self.dataset, column, self.queues))        
        save_task = asyncio.create_task(self.save_to_disk(dataset, dst_path, models))
        await asyncio.gather(producer_task, save_task, *consumer_tasks.values())
        self.save_to_disk(dataset, dst_path, models)
        return None 

    
    async def load_checkpoints(self, dataset, split, dst_path, models):
        if "new_dataset" not in list(dir(self)):
            self.new_dataset = None
        if "all_cid_list" not in list(dir(self)):
            self.all_cid_list = {}
        if "all_cid_set" not in list(dir(self)):
            self.all_cid_set = {}
        for model in models:
            if model not in list(self.index.keys()):
                self.index[model] = None
        if self.new_dataset is None or isinstance(self.new_dataset, dict):
            new_dataset_dst_path = os.path.join(dst_path, "ipfs_" + dataset.replace("/","___") + ".parquet")
            if os.path.isfile(new_dataset_dst_path):
                self.new_dataset = load_dataset('parquet', data_files=new_dataset_dst_path)[split]
            if os.path.exists(os.path.join(dst_path, "checkpoints")):
                ls_checkpoints = os.listdir(os.path.join(dst_path, "checkpoints"))
                new_dataset_shards = [os.path.join(dst_path, "checkpoints", x) for x in ls_checkpoints if "ipfs_" + dataset.replace("/", "___") + "_shard" in x and "_cids" not in x]
                if "new_dataset" not in list(self.all_cid_list.keys()):
                    self.all_cid_list["new_dataset"] = []
                if "new_dataset" not in list(self.all_cid_set.keys()):
                    self.all_cid_set["new_dataset"] = set()
                for shard in new_dataset_shards:
                    if os.path.exists(shard.replace(".parquet","")+"_cids.parquet"):
                        tmp_new_dataset_cids = load_dataset('parquet', data_files=shard.replace(".parquet","")+"_cids.parquet")["train"]
                        self.all_cid_list["new_dataset"] += list(tmp_new_dataset_cids["cids"])
                        self.all_cid_set["new_dataset"] = self.all_cid_set["new_dataset"].union(set(tmp_new_dataset_cids["cids"]))
                        del tmp_new_dataset_cids
                    else:
                        new_dataset_shard = load_dataset('parquet', data_files=shard)["train"]
                        tmp_new_dataset_cids = new_dataset_shard.map(lambda x: {"cid": x["items"]["cid"]})["cids"]
                        self.all_cid_list["new_dataset"] += list(tmp_new_dataset_cids)
                        self.all_cid_set["new_dataset"] = self.all_cid_set["new_dataset"].union(set(tmp_new_dataset_cids))
                        tmp_new_dataset_cid_dataset = datasets.Dataset.from_dict({"cids": tmp_new_dataset_cids})
                        tmp_new_dataset_cid_dataset.to_parquet(shard.replace(".parquet","")+"_cids.parquet")
                        del new_dataset_shard
                        del tmp_new_dataset_cids
                        del tmp_new_dataset_cid_dataset
                if self.new_dataset is None or isinstance(self.new_dataset, dict):
                    if len(new_dataset_shards) > 0:
                        self.new_dataset = load_dataset('parquet', data_files=new_dataset_shards)[split]
                    else:
                        columns = self.dataset.column_names
                        columns.append("cid")
                        self.new_dataset = datasets.Dataset.from_dict({key: [] for key in columns })
        for model in models:
            if model not in list(self.index.keys()):
                self.index[model] = None
            if model not in list(self.all_cid_list.keys()):
                self.all_cid_list[model] = []
            if model not in list(self.all_cid_set.keys()):
                self.all_cid_set[model] = set()
            model_dst_path = dst_path + "/" + model.replace("/","___") + ".parquet"
            if os.path.isfile(model_dst_path):
                self.caches[model] = {"items" : []}
                self.index[model] = load_dataset('parquet', data_files=model_dst_path, streaming=True)[split]
            if os.path.exists(os.path.join(dst_path, "checkpoints")):
                ls_checkpoints = os.listdir(os.path.join(dst_path, "checkpoints"))
                this_model_shards = [os.path.join(dst_path, "checkpoints", x)  for x in ls_checkpoints if model.replace("/", "___") + "_shard" in x and "_cids" not in x]
                for shard in this_model_shards:
                    if os.path.exists(shard.replace(".parquet","")+"_cids.parquet"):
                        tmp_model_cids = load_dataset('parquet', data_files=shard.replace(".parquet","")+"_cids.parquet")["train"]
                        self.all_cid_list[model] += list(tmp_model_cids["cids"])
                        self.all_cid_set[model] = self.all_cid_set[model].union(set(tmp_model_cids["cids"]))
                        del tmp_model_cids
                    else:
                        this_model_shard = load_dataset('parquet', data_files=shard)[split]
                        tmp_model_cids = this_model_shard.map(lambda x: {"cid": x["items"]["cid"]})["cid"]
                        self.all_cid_list[model] += list(tmp_model_cids)
                        self.all_cid_set[model] = self.all_cid_set[model].union(set(tmp_model_cids))
                        tmp_model_cid_dataset = datasets.Dataset.from_dict({"cids": tmp_model_cids})
                        tmp_model_cid_dataset.to_parquet(shard.replace(".parquet","")+"_cids.parquet")
                        del this_model_shard
                        del tmp_model_cids
                        del tmp_model_cid_dataset
                if model not in list(self.index.keys()) or self.index[model] is None or isinstance(self.index[model], dict):
                    if len(this_model_shards) > 0:
                        self.index[model] = load_dataset('parquet', data_files=this_model_shards)[split]
                    else:
                        self.index[model] = datasets.Dataset.from_dict({"cid": [], "embedding": [] })
                ls_chunks = os.listdir(os.path.join(dst_path, "checkpoints", "sparse_chunks"))
                for chunk in ls_chunks:
                    chunk_cid = chunk.replace(".parquet","")
                    if chunk.replace(".parquet","") not in self.cid_chunk_set:
                        self.cid_chunk_set.add(chunk_cid)
                        self.cid_chunk_list.append(chunk_cid)
                del ls_chunks
                del this_model_shards
                del ls_checkpoints
                del new_dataset_shards
        self.cid_set = set.intersection(*self.all_cid_set.values())
        return None
    
    async def combine_checkpoints(self, dataset, split, columns, dst_path, models):
        await self.load_checkpoints(dataset, split, dst_path, models)
        columns = self.new_dataset.column_names
        self.new_dataset_combined = datasets.Dataset.from_dict({key: [] for key in columns })
        self.embedding_datasets = {}
        count_cids = 0
        len_cids = len(self.cid_set)
        for model in models:
            self.embedding_datasets[model] = datasets.Dataset.from_dict({key: [] for key in columns })
        
        for cid in self.cid_set:
            new_dataset_index = self.all_cid_list["new_dataset"].index(cid)
            new_dataset_item = self.new_dataset.select([new_dataset_index])[0]
            self.new_dataset_combined = self.new_dataset_combined.add_item(new_dataset_item["items"])
            for model in models:
                if model in list(self.index.keys()):
                    embedding_dataset_index = self.all_cid_list[model].index(cid)
                    embedding_dataset_item = self.index[model].select([embedding_dataset_index])[0]
                    self.embedding_datasets[model] = self.embedding_datasets[model].add_item(embedding_dataset_item["items"])
            count_cids += 1
            if count_cids % 1000 == 0:
                print("Sorted " + str(count_cids) + " of " + str(len_cids) + " cids")
        self.new_dataset_combined.to_parquet(os.path.join(dst_path, dataset.replace("/","___") + ".parquet"))
        for model in models:
            self.embedding_datasets[model].to_parquet(os.path.join(dst_path, dataset.replace("/","___") + "_" + model.replace("/","___") + ".parquet"))
        return None

    async def kmeans_cluster_split(self, dataset, split, columns, dst_path, models, max_splits):
        if self.new_dataset is None or isinstance(self.new_dataset, dict):
            await self.load_checkpoints(dataset, split, dst_path, models)
        #deduplicate self.new_dataset
        self.unique_dataset = self.new_dataset.map(lambda x: {"cid": x["cid"], "items": x["items"]})
        dataset_sizes = {}
        dataset_sizes["unique_dataset"] = len(self.unique_dataset)
        for model in models:
            if model in list(self.index.keys()):
                dataset_sizes[model] = len(self.index[model])
        
        # Initialize variables
        centroids = []
        embeddings = []
        
        # Convert embeddings to numpy array
        embeddings_np = np.array(embeddings)
        
        # Perform KMeans clustering using faiss
        kmeans = faiss.Kmeans(d=embeddings_np.shape[1], k=max_splits, niter=300, verbose=True)
        kmeans.train(embeddings_np)

        # Get centroids
        centroids = kmeans.centroids

        # Save centroids to disk
        centroids_dataset = datasets.Dataset.from_dict({"centroids": centroids.tolist()})
        centroids_dataset.to_parquet(os.path.join(dst_path, dataset.replace("/", "___") + "_centroids.parquet"))

        return None
    