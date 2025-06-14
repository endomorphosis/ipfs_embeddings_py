import torch
import requests
import json
import os
import sys
import asyncio
import random
import aiohttp
from aiohttp import ClientSession, ClientTimeout
from transformers import AutoTokenizer
from transformers import AutoModel
class ipfs_accelerate_py:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        if "test_ipfs_embeddings_py" not in globals():
            import test.ipfs.test_ipfs_embeddings
            from test.ipfs.test_ipfs_embeddings import test.ipfs.test_ipfs_embeddings_py
            self.test_ipfs_embeddings = test_ipfs_embeddings_py(resources, metadata)
        else:
            self.test_ipfs_embeddings = test_ipfs_embeddings(resources, metadata) 
        if "install_depends_py" not in globals():
            import install_depends
            from install_depends import install_depends_py
            self.install_depends = install_depends_py(resources, metadata) 
        else:
            self.install_depends = install_depends_py(resources, metadata)
        self.tei_endpoints = {}
        self.openvino_endpoints = {}
        self.libp2p_endpoints = {}
        self.local_endpoints = {}
        self.endpoint_status = {}
        self.endpoint_handler = {}
        self.batch_sizes = {}
        self.tokenizer = {}
        self.queues = {}
        self.local_endpoints = {}
        self.tei_endpoints = {}
        self.openvino_endpoints = {}
        self.libp2p_endpoints = {}
        self.endpoint_types = ["tei_endpoints", "openvino_endpoints", "libp2p_endpoints", "local_endpoints"]
        self.add_endpoint = self.add_endpoint
        self.rm_endpoint = self.rm_endpoint
        self.init_endpoints = self.init_endpoints
        self.get_https_endpoint = self.get_https_endpoint
        self.get_libp2p_endpoint = self.get_libp2p_endpoint
        self.request_tei_endpoint = self.request_tei_endpoint
        self.request_libp2p_endpoint = self.request_libp2p_endpoint
        self.request_openvino_endpoint = self.request_openvino_endpoint
        self.request_local_endpoint = self.request_local_endpoint
        self.test_tei_https_endpoint = self.test_tei_https_endpoint
        self.test_libp2p_endpoint = self.test_libp2p_endpoint
        self.test_openvino_endpoint = self.test_openvino_endpoint
        self.test_local_endpoint = self.test_local_endpoint
        return None
    
    async def test_hardware(self):
        return await self.test_ipfs_embeddings.test_hardware()

    async def init_endpoints(self, models=None, endpoint_list=None):
        for endpoint_type in self.endpoint_types:
            if endpoint_type in resources.keys():
                for endpoint_info in resources[endpoint_type]:
                    model, endpoint, context_length = endpoint_info
                    await self.add_endpoint(model, endpoint, context_length, endpoint_type)    
        if "hwtest" not in dir(self):
            hwtest = await self.test_hardware()
            self.hwtest = hwtest
        for model in models:
            if model not in self.queues:
                self.queues[model] = {}
        if endpoint_list is None:    
            endpoints = self.get_endpoints(model)
            local = self.get_endpoints(model, "local")
            openvino = self.get_endpoints(model, "openvino")
            libp2p = self.get_endpoints(model, "libp2p")
            tei = self.get_endpoints(model, "tei")
            cuda = self.get_endpoints(model, "cuda")
            endpoints =  { "tei" : tei , "local" : local , "openvino": openvino , "libp2p": libp2p , "cuda": cuda }
            endpoints_set = set(endpoints["tei"] + endpoints["local"] + endpoints["openvino"] + endpoints["libp2p"] + endpoints["cuda"])
        else:
            endpoints = endpoint_list
            self.endpoints = endpoints
            endpoints_list = [ k for k in endpoints.keys() ]
            self.endpoints_list = endpoints_list
            self.endpoints = { k : v for k, v in enumerate(endpoints_list) }
            endpoints_set = set(endpoints_list)
            self.endpoint_set = endpoints_set
            local = [ endpoint for endpoint in endpoints["local_endpoints"] if "local" in endpoint or "cpu" in endpoint or "cuda" in endpoint or "openvino" in endpoint or "llama_cpp" in endpoint or "ipex" in endpoint]
            openvino = [endpoint for endpoint in endpoints["openvino_endpoints"] ]
            libp2p = []
            tei = [endpoint for endpoint in endpoints["tei_endpoints"]]
            pass
        if type(endpoint_list) == list:
            self.endpoints = { k : v for k, v in enumerate(endpoint_list) }
            endpoints = endpoint_list
            self.endpoint_list = endpoint_list
            endpoints_set = set(endpoints)
            self.endpoint_set = endpoints_set
        if type(endpoint_list) == dict:
            self.endpoints = endpoint_list
            endpoints_list = list(endpoint_list.keys())
            endpoints_set = set(endpoints_list)
            self.endpoint_set = endpoints_set
        if not endpoints_set:
            raise ValueError("No endpoints available for model " + model)
        cuda_test = self.hwtest["cuda"]
        openvino_test = self.hwtest["openvino"]
        llama_cpp_test = self.hwtest["llama_cpp"]
        ipex_test = self.hwtest["ipex"]
        cpus = os.cpu_count()
        cuda = torch.cuda.is_available()
        gpus = torch.cuda.device_count()
        for model in models:
            if model not in self.tokenizer:
                self.tokenizer[model] = {}
            if model not in self.local_endpoints:
                self.local_endpoints[model] = {}
            if model not in self.queues:    
                self.queues[model] = {}
            if model not in self.batch_sizes:
                self.batch_sizes[model] = {}
            if "cpu" not in self.local_endpoints[model]:
                self.local_endpoints[model]["cpu"] = ""
            if "cpu" not in self.queues[model]:
                self.queues[model]["cpu"] = ""
            if "cpu" not in self.batch_sizes[model]:
                self.batch_sizes[model]["cpu"] = 1
        for model in models:
            if cuda and gpus > 0:
                if cuda_test and type(cuda_test) != ValueError:
                    self.local_endpoints[model] = {"cuda:" + str(gpu) : None for gpu in range(gpus) } if gpus > 0 else {"cpu": None}
                    for gpu in range(gpus):
                        self.tokenizer[model]["cuda:" + str(gpu)] = AutoTokenizer.from_pretrained(model, device='cuda:' + str(gpu), use_fast=True)
                        self.local_endpoints[model]["cuda:" + str(gpu)] = AutoModel.from_pretrained(model).to("cuda:" + str(gpu))
                        torch.cuda.empty_cache()
                        self.queues[model]["cuda:" + str(gpu)] = asyncio.Queue(64)
                        batch_size = await self.max_batch_size(model, "cuda:" + str(gpu))
                        self.local_endpoints[model]["cuda:" + str(gpu)] = batch_size
                        self.batch_sizes[model]["cuda:" + str(gpu)] = batch_size
                        self.endpoint_handler[(model, "cuda:" + str(gpu))] = ""
                        # consumer_tasks[(model, "cuda:" + str(gpu))] = asyncio.create_task(self.chunk_consumer(batch_size, model, "cuda:" + str(gpu)))
            if len(local) > 0 and cpus > 0:
                all_test_types = [ type(openvino_test), type(llama_cpp_test), type(ipex_test)]
                all_tests_ValueError = all(x is ValueError for x in all_test_types)
                all_tests_none = all(x is None for x in all_test_types)
                if all_tests_ValueError or all_tests_none:  
                    self.local_endpoints[model]["cpu"] = AutoModel.from_pretrained(model).to("cpu")
                    self.queues[model]["cpu"] = asyncio.Queue(4)
                    self.endpoint_handler[(model, "cpu")] = ""
                    # consumer_tasks[(model, "cpu")] = asyncio.create_task(self.chunk_consumer( 1, model, "cpu"))
                elif openvino_test and type(openvino_test) != ValueError:
                    ov_count = 0
                    for endpoint in local:
                        if "openvino" in endpoint[1]:
                            endpoint_name = "openvino:"+str(ov_count)
                            batch_size = 0
                            if model not in self.batch_sizes:
                                self.batch_sizes[model] = {}
                            if model not in self.queues:
                                self.queues[model] = {}
                            if endpoint not in list(self.batch_sizes[model].keys()):
                                batch_size = await self.max_batch_size(model, endpoint)
                                self.batch_sizes[model][endpoint_name] = batch_size
                            if self.batch_sizes[model][endpoint_name] > 0:
                                self.queues[model][endpoint_name] = asyncio.Queue(64)
                                self.endpoint_handler[(model, endpoint_name)] = ""
                                # consumer_tasks[(model, endpoint_name )] = asyncio.create_task(self.chunk_consumer(batch_size, model, endpoint_name))
                            ov_count = ov_count + 1
                elif llama_cpp_test and type(llama_cpp_test) != ValueError:
                    llama_count = 0
                    for endpoint in local:
                        if "llama_cpp" in endpoint:
                            endpoint_name = "llama:"+str(ov_count)
                            batch_size = 0                            
                            if model not in self.batch_sizes:
                                self.batch_sizes[model] = {}
                            if model not in self.queues:
                                self.queues[model] = {}
                            if endpoint not in list(self.batch_sizes[model].keys()):
                                batch_size = await self.max_batch_size(model, endpoint)
                                self.batch_sizes[model][endpoint] = batch_size
                            if self.batch_sizes[model][endpoint] > 0:
                                self.queues[model][endpoint] = asyncio.Queue(64)
                                self.endpoint_handler[(model, endpoint_name)] = ""
                                # consumer_tasks[(model, endpoint)] = asyncio.create_task(self.chunk_consumer(batch_size, model, endpoint_name))
                            llama_count = llama_count + 1
                elif ipex_test and type(ipex_test) != ValueError:
                    ipex_count = 0
                    for endpoint in local:
                        if "ipex" in endpoint:
                            endpoint_name = "ipex:"+str(ipex_count)
                            batch_size = 0
                            if model not in self.batch_sizes:
                                self.batch_sizes[model] = {}
                            if model not in self.queues:
                                self.queues[model] = {}
                            if endpoint not in list(self.batch_sizes[model].keys()):
                                batch_size = await self.max_batch_size(model, endpoint)
                                self.batch_sizes[model][endpoint] = batch_size
                            if self.batch_sizes[model][endpoint] > 0:
                                self.queues[model][endpoint] = asyncio.Queue(64)
                                self.endpoint_handler[(model, endpoint_name)] = ""
                                # consumer_tasks[(model, endpoint)] = asyncio.create_task(self.chunk_consumer(self.queues[model][endpoint], column, batch_size, model, endpoint))
                            ipex_count = ipex_count + 1
            if "openvino_endpoints" in list(self.endpoints.keys()):
                if len(self.endpoints["openvino_endpoints"]) > 0 :
                    for endpoint in self.endpoints["openvino_endpoints"]:
                        batch_size = 0
                        if model not in self.batch_sizes:
                            self.batch_sizes[model] = {}
                        if model not in self.queues:
                            self.queues[model] = {}
                        if endpoint not in list(self.batch_sizes[model].keys()):
                            batch_size = await self.max_batch_size(model, endpoint)
                            self.batch_sizes[model][endpoint] = batch_size
                        if self.batch_sizes[model][endpoint] > 0:
                            self.queues[model][endpoint] = asyncio.Queue(64)  # Unbounded queue
                            self.endpoint_handler[(model, endpoint)] = ""
                            # consumer_tasks[(model, endpoint)] = asyncio.create_task(self.chunk_consumer(self.queues[model][endpoint], column, batch_size, model, endpoint))
            if "tei_endpoints" in list(self.endpoints.keys()):
                if len(self.endpoints["tei_endpoints"]) > 0:
                    for endpoint in self.endpoints["tei_endpoints"]:
                        this_model = endpoint[0]
                        this_endpoint = endpoint[1]
                        context_length = endpoint[2]
                        batch_size = 0
                        if this_model not in self.batch_sizes:
                            self.batch_sizes[this_model] = {}
                        if this_model not in self.queues:
                            self.queues[model] = {}
                        if endpoint not in list(self.batch_sizes[model].keys()):
                            batch_size = await self.max_batch_size(model, endpoint)
                            self.batch_sizes[model][this_endpoint] = batch_size
                        if self.batch_sizes[model][this_endpoint] > 0:
                            self.queues[model][this_endpoint] = asyncio.Queue(64)  # Unbounded queue
                            self.endpoint_handler[(model, this_endpoint)] = ""
                            # consumer_tasks[(model, endpoint)] = asyncio.create_task(self.chunk_consumer(batch_size, model, endpoint)) 
            if "libp2p_endpoints" in list(self.endpoints.keys()):
                if len(self.endpoints["libp2p_endpoints"]) > 0:
                    for endpoint in self.endpoints["libp2p_endpoints"]:
                        batch_size = 0
                        if model not in self.batch_sizes:
                            self.batch_sizes[model] = {}
                        if model not in self.queues:
                            self.queues[model] = {}
                        if endpoint not in list(self.batch_sizes[model].keys()):
                            batch_size = await self.max_batch_size(model, endpoint)
                            self.batch_sizes[model][endpoint] = batch_size
                        if self.batch_sizes[model][endpoint] > 0:
                            self.queues[model][endpoint] = asyncio.Queue(64)
        return None

    
    def test_tei_https_endpoint(self, model, endpoint):
        if model in self.tei_endpoints and endpoint in self.tei_endpoints[model]:
            return True
        return False

    def test_libp2p_endpoint(self, model, endpoint):
        if model in self.libp2p_endpoints and endpoint in self.libp2p_endpoints[model]:
            return True
        return False
    
    def test_openvino_endpoint(self, model, endpoint):
        if model in self.openvino_endpoints and endpoint in self.openvino_endpoints[model]:
            return True
        return False
    
    def test_local_endpoint(self, model, endpoint):
        if model in self.local_endpoints and endpoint in self.local_endpoints[model]:
            return True
        return False

    def get_https_endpoint(self, model):
        if model in self.tei_endpoints:
            return self.tei_endpoints[model]
        return None

    def get_libp2p_endpoint(self, model):
        if model in self.libp2p_endpoints:
            return self.libp2p_endpoints[model]
        return None

    def request_tei_endpoint(self, model, endpoint, endpoint_type, batch):
        incoming_batch_size = len(batch)
        endpoint_batch_size = 0
        if endpoint in self.endpoint_status:
            endpoint_batch_size = self.endpoint_status[endpoint]
        elif endpoint_type == None:
            for endpoint_type in self.endpoint_types:
                if endpoint_type in self.__dict__.keys():
                    if model in self.__dict__[endpoint_type]:
                        for endpoint in self.__dict__[endpoint_type][model]:
                            endpoint_batch_size = self.endpoint_status[endpoint]
                            if self.endpoint_status[endpoint] >= incoming_batch_size:
                                return endpoint
                    else:
                        if incoming_batch_size > endpoint_batch_size:
                            return ValueError("Batch size too large")
                        else:
                            return None
                else:
                    pass
        else:
            if model in self.__dict__[endpoint_type]:
                for endpoint in self.__dict__[endpoint_type][model]:
                    endpoint_batch_size = self.endpoint_status[endpoint]
                    if self.endpoint_status[endpoint] >= incoming_batch_size:
                        return endpoint
                    else:
                        if incoming_batch_size > endpoint_batch_size:
                            return ValueError("Batch size too large")
                        else:
                            return None
            else:
                return None
                
        if incoming_batch_size > endpoint_batch_size:
            return ValueError("Batch size too large")
        else:
            if model in self.endpoints:
                for endpoint in self.tei_endpoints[model]:
                    if self.endpoint_status[endpoint] >= incoming_batch_size:
                        return endpoint
            return None
    
    async def request_openvino_endpoint(self, model, batch_size):
        if model in self.openvino_endpoints:
            for endpoint in self.openvino_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_llama_cpp_endpoint(self, model, batch_size):
        if model in self.llama_cpp_endpoints:
            for endpoint in self.llama_cpp_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_libp2p_endpoint(self, model, batch_size):
        if model in self.libp2p_endpoints:
            for endpoint in self.libp2p_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_local_endpoint(self, model, batch_size):
        if model in self.local_endpoints:
            for endpoint in self.local_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None

    async def make_local_request(self, model, endpoint, endpoint_type, data):
        device = torch.device(endpoint)
        inputs = self.tokenizer[model][endpoint](data, return_tensors="pt", padding=True, truncation=True).to(device)
        self.local_endpoints[model][endpoint].to(device).eval()
        with torch.no_grad():
            outputs = self.local_endpoints[model][endpoint](**inputs)
            query_response = outputs.last_hidden_state.mean(dim=1).tolist()  # Use mean of token embeddings
            results = query_response  # Return the entire batch of results
            del inputs, outputs  # Unallocate inputs and outputs
            torch.cuda.synchronize()  # Ensure all operations are complete
            torch.cuda.empty_cache()  # Free up GPU memory
        # self.local_endpoints[model][endpoint].to('cpu')  # Move model back to CPU
        torch.cuda.empty_cache()  # Free up GPU memory again
        return results

    async def add_endpoint(self, model, endpoint, context_length, endpoint_type):
        if endpoint_type in self.endpoint_types:
            success = False
            try:
                if endpoint_type not in list(dir(self)):
                    self.__dict__[endpoint_type]= {}
                if model not in list(self.__dict__[endpoint_type].keys()):
                    self.__dict__[endpoint_type][model] = {}
                if endpoint not in list(self.__dict__[endpoint_type][model].keys()):
                    self.__dict__[endpoint_type][model][endpoint] = context_length
                self.endpoint_status[endpoint] = context_length
                success = True
            except Exception as e:
                print(e)
                pass
            return success        
        return None
    
    async def rm_endpoint(self, model, endpoint, endpoint_type):
        if endpoint_type in self.endpoint_types:
            success = False
            try:
                if model in self.__dict__[endpoint_type] and endpoint in self.__dict__[endpoint_type][model]:
                    del self.__dict__[endpoint_type][model][endpoint]
                if endpoint in self.endpoint_status:
                    del self.endpoint_status[endpoint]
                success = True
            except Exception as e:
                print(e)
                pass
            return success
        return None
    
    async def max_batch_size(self, model, endpoint=None, endpoint_type=None ):
        embed_fail = False
        exponent = 0
        batch = []
        token_length_size = 0
        batch_size = 2**exponent
        if endpoint_type is None:
            this_model = None
            this_endpoint = None
            this_context_length = None
            if "/embed" in endpoint:
                endpoint_type = "tei_endpoints"
            elif "/infer" in endpoint:
                endpoint_type = "openvino_endpoints"
            elif "http" in endpoint:
                endpoint_type = "tei_endpoints"
            elif "cuda" in endpoint or "cpu" in endpoint or "local" in endpoint:
                endpoint_type = "local_endpoints"
            elif "libp2p" in endpoint:
                endpoint_type = "libp2p_endpoints"
            if endpoint_type is None:
                print('Endpoint not found')
                return 0
            else:
                pass
                  
        for this_endpoint in self.endpoints[endpoint_type]:
            if "cuda" in this_endpoint[1] or "cpu" in this_endpoint[1] or "local" in this_endpoint[1]:
                this_endpoint_index = self.endpoints[endpoint_type].index(this_endpoint)
                token_length_size = round(self.endpoints["local_endpoints"][this_endpoint_index][2] * 0.99)
            elif model is this_endpoint[0]:
                this_endpoint_index = self.endpoints[endpoint_type].index(this_endpoint)
                token_length_size = round(self.endpoints[endpoint_type][this_endpoint_index][2] * 0.99) 
        
        test_tokens = []
        if model not in self.tokenizer.keys():
            self.tokenizer[model] = {}
        if "cpu" not in self.tokenizer[model].keys():
            self.tokenizer[model]["cpu"] = AutoTokenizer.from_pretrained(model, device='cpu')
        find_token_str = str("z")
        find_token_int = self.tokenizer[model]["cpu"].encode(find_token_str)
        if len(find_token_int) == 3:
            find_token_int = find_token_int[1]
        elif len(find_token_int) == 2:
            find_token_int = find_token_int[1]
        elif len(find_token_int) == 1:
            find_token_int = find_token_int[0]
        for i in range(token_length_size):
             test_tokens.append(find_token_int)
        test_text = self.tokenizer[model]["cpu"].decode(test_tokens)
        if endpoint is None:
            endpoint = self.choose_endpoint(model)
        while not embed_fail:
            test_batch = []
            for i in range(batch_size):
                test_batch.append(test_text)
            parsed_knn_embeddings = None
            embeddings = None
            request_knn_results = None
            try:
                request_knn_results = await self.request_knn(test_batch, model, endpoint, endpoint_type)
            except Exception as e:
                try:
                    embeddings = await self.index_knn(test_batch, model, endpoint)
                except Exception as e:
                        pass
            if request_knn_results != None and parsed_knn_embeddings == None:
                parsed_knn_embeddings = await self.parse_knn(request_knn_results, model, endpoint, endpoint_type)
            if parsed_knn_embeddings is not None:
               embeddings = parsed_knn_embeddings
            
        self.endpoint_status[endpoint] = 2**(exponent-1)
        if exponent == 0:
            return 1
        else:
            return 2**(exponent-1)
    
    async def request_openvino_endpoint(self, model, batch_size):
        if model in self.openvino_endpoints:
            for endpoint in self.openvino_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_llama_cpp_endpoint(self, model, batch_size):
        if model in self.llama_cpp_endpoints:
            for endpoint in self.llama_cpp_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_libp2p_endpoint(self, model, batch_size):
        if model in self.libp2p_endpoints:
            for endpoint in self.libp2p_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None
    
    async def request_local_endpoint(self, model, batch_size):
        if model in self.local_endpoints:
            for endpoint in self.local_endpoints[model]:
                if self.endpoint_status[endpoint] >= batch_size:
                    return endpoint
        return None

    def test_tei_https_endpoint(self, model, endpoint):
        if model in self.tei_endpoints and endpoint in self.tei_endpoints[model]:
            return True
        return False

    def test_libp2p_endpoint(self, model, endpoint):
        if model in self.libp2p_endpoints and endpoint in self.libp2p_endpoints[model]:
            return True
        return False
    
    def test_openvino_endpoint(self, model, endpoint):
        if model in self.openvino_endpoints and endpoint in self.openvino_endpoints[model]:
            return True
        return False
    
    def test_local_endpoint(self, model, endpoint):
        if model in self.local_endpoints and endpoint in self.local_endpoints[model]:
            return True
        return False

    def get_https_endpoint(self, model):
        if model in self.tei_endpoints:
            return self.tei_endpoints[model]
        return None

    def get_libp2p_endpoint(self, model):
        if model in self.libp2p_endpoints:
            return self.libp2p_endpoints[model]
        return None

    def request_tei_endpoint(self, model, endpoint, endpoint_type, batch):
        incoming_batch_size = len(batch)
        endpoint_batch_size = 0
        if endpoint in self.endpoint_status:
            endpoint_batch_size = self.endpoint_status[endpoint]
        elif endpoint_type == None:
            for endpoint_type in self.endpoint_types:
                if endpoint_type in self.__dict__.keys():
                    if model in self.__dict__[endpoint_type]:
                        for endpoint in self.__dict__[endpoint_type][model]:
                            endpoint_batch_size = self.endpoint_status[endpoint]
                            if self.endpoint_status[endpoint] >= incoming_batch_size:
                                return endpoint
                    else:
                        if incoming_batch_size > endpoint_batch_size:
                            return ValueError("Batch size too large")
                        else:
                            return None
                else:
                    pass
        else:
            if model in self.__dict__[endpoint_type]:
                for endpoint in self.__dict__[endpoint_type][model]:
                    endpoint_batch_size = self.endpoint_status[endpoint]
                    if self.endpoint_status[endpoint] >= incoming_batch_size:
                        return endpoint
                    else:
                        if incoming_batch_size > endpoint_batch_size:
                            return ValueError("Batch size too large")
                        else:
                            return None
            else:
                return None
                
        if incoming_batch_size > endpoint_batch_size:
            return ValueError("Batch size too large")
        else:
            if model in self.endpoints:
                for endpoint in self.tei_endpoints[model]:
                    if self.endpoint_status[endpoint] >= incoming_batch_size:
                        return endpoint
            return None
    
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

    async def make_post_request_libp2p(self, endpoint, data):
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
                    
    async def make_post_request_openvino(self, endpoint, data):
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
 
    async def choose_endpoint(self, model, endpoint_type=None):
        if endpoint_type != None:
            this_endpoints = self.get_endpoints(model, endpoint_type)
        else:
            tei_endpoints = self.get_endpoints(model, endpoint_type="tei")
            libp2p_endpoints = self.get_endpoints(model, endpoint_type="libp2p")
            openvino_endpoints = self.get_endpoints(model, endpoint_type="openvino")
            local_endpoints = self.get_endpoints(model, endpoint_type="local")
            filtered_libp2p_endpoints = {k: v for k, v in self.endpoint_status.items() if v >= 1 and libp2p_endpoints is not None and k in list(libp2p_endpoints.keys())}
            filtered_tei_endpoints = {k: v for k, v in self.endpoint_status.items() if v >= 1 and tei_endpoints is not None and k in list(tei_endpoints.keys())}
            filtered_openvino_endpoints = {k: v for k, v in self.endpoint_status.items() if v >= 1 and openvino_endpoints is not None and k in list(openvino_endpoints.keys())}
            filtered_local_endpoints = {k: v for k, v in self.endpoint_status.items() if v >= 1 and local_endpoints is not None and k in list(local_endpoints.keys())}
            if not filtered_tei_endpoints and not filtered_libp2p_endpoints and not filtered_openvino_endpoints and not filtered_local_endpoints:
                return None
            else:
                this_endpoint = None
                if len(list(filtered_local_endpoints.keys())) > 0:
                    this_endpoint = random.choice(list(filtered_local_endpoints.keys()))
                if len(list(filtered_tei_endpoints.keys())) > 0:
                    this_endpoint = random.choice(list(filtered_tei_endpoints.keys()))
                if len(list(filtered_openvino_endpoints.keys())) > 0:
                    this_endpoint = random.choice(list(filtered_openvino_endpoints.keys()))
                elif len(list(filtered_libp2p_endpoints.keys())) > 0:
                    this_endpoint = random.choice(list(filtered_libp2p_endpoints.keys()))
                print("chosen endpoint for " + model + " is " + this_endpoint)
                return this_endpoint


    async def get_endpoints(self, model, endpoint_type=None):
        if endpoint_type is None:
            endpoints_dict = self.tei_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoints_dict if self.endpoint_status.get(endpoint, 0) >= 1]
        if endpoint_type == "tei":
            endpoints_dict = self.tei_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoints_dict if self.endpoint_status.get(endpoint, 0) >= 1]
        if endpoint_type == "openvino":
            endpoints_dict = self.openvino_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoints_dict if self.endpoint_status.get(endpoint, 0) >= 1]
        if endpoint_type == "libp2p":
            endpoints_dict = self.libp2p_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoints_dict if self.endpoint_status.get(endpoint, 0) >= 1]
        if endpoint_type == "local":
            endpoints_dict = self.local_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoints_dict if self.endpoint_status.get(endpoint, 0) >= 1]
        if endpoint_type == "cuda":
            endpoint_dict = self.local_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in endpoint_dict if "cuda" in endpoint and self.endpoint_status.get(endpoint, 0) >= 1]
        else:
            all_endpoints_dict = self.tei_endpoints.get(model, {}) + self.libp2p_endpoints.get(model, {}) + self.openvino_endpoints.get(model, {}) + self.local_endpoints.get(model, {})
            filtered_endpoints = [endpoint for endpoint in all_endpoints_dict if self.endpoint_status.get(endpoint, 0) >= 1]
        return filtered_endpoints
    
    async def async_generator(self, iterable):
        for item in iterable:
            yield item
    
    async def __test__(self, resources, metadata):
        results = {}
        test_ipfs_accelerate = self.__init__(resources, metadata)
        test_ipfs_accelerate_init = await self.init_endpoints( metadata['models'], resources)
        results = {"test_ipfs_accelerate_init": test_ipfs_accelerate_init, "test_ipfs_accelerate": test_ipfs_accelerate}
        return results

ipfs_accelerate_py = ipfs_accelerate_py

if __name__ == "__main__":
    metadata = {
        "dataset": "TeraflopAI/Caselaw_Access_Project",
        "namespace": "TeraflopAI/Caselaw_Access_Project",
        "column": "text",
        "split": "train",
        "models": [
            "thenlper/gte-small",
            # "Alibaba-NLP/gte-large-en-v1.5",
            # "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        ],
        "chunk_settings": {
            "chunk_size": 512,
            "n_sentences": 8,
            "step_size": 256,
            "method": "fixed",
            "embed_model": "thenlper/gte-small",
            "tokenizer": None
        },
        "dst_path": "/storage/teraflopai/tmp",
    }
    resources = {
        "local_endpoints": [
            ["thenlper/gte-small", "cpu", 512],
            ["Alibaba-NLP/gte-large-en-v1.5", "cpu", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "cpu", 32768],
            ["thenlper/gte-small", "cuda:0", 512],
            ["Alibaba-NLP/gte-large-en-v1.5", "cuda:0", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "cuda:0", 32768],
            ["thenlper/gte-small", "cuda:1", 512],
            ["Alibaba-NLP/gte-large-en-v1.5", "cuda:1", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "cuda:1", 32768],
            ["thenlper/gte-small", "openvino", 512],
            ["Alibaba-NLP/gte-large-en-v1.5", "openvino", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "openvino", 32768],
            ["thenlper/gte-small", "llama_cpp", 512],
            ["Alibaba-NLP/gte-large-en-v1.5", "llama_cpp", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "llama_cpp", 32768],
            ["thenlper/gte-small", "ipex", 512],
            ["Alibaba-NLP/gte-large-en-v1.5", "ipex", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "ipex", 32768],
        ],
        "openvino_endpoints": [
            # ["neoALI/bge-m3-rag-ov", "https://bge-m3-rag-ov-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-rag-ov/infer", 4095],
            # ["neoALI/bge-m3-rag-ov", "https://bge-m3-rag-ov-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-rag-ov/infer", 4095],
            # ["neoALI/bge-m3-rag-ov", "https://bge-m3-rag-ov-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-rag-ov/infer", 4095],
            # ["neoALI/bge-m3-rag-ov", "https://bge-m3-rag-ov-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-rag-ov/infer", 4095],
            # ["aapot/bge-m3-onnx", "https://bge-m3-onnx0-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx0/infer", 1024],
            # ["aapot/bge-m3-onnx", "https://bge-m3-onnx1-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx1/infer", 1024],
            # ["aapot/bge-m3-onnx", "https://bge-m3-onnx2-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx2/infer", 1024],
            # ["aapot/bge-m3-onnx", "https://bge-m3-onnx3-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx3/infer", 1024],
            # ["aapot/bge-m3-onnx", "https://bge-m3-onnx4-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx4/infer", 1024],
            # ["aapot/bge-m3-onnx", "https://bge-m3-onnx5-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx5/infer", 1024],
            # ["aapot/bge-m3-onnx", "https://bge-m3-onnx6-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx6/infer", 1024],
            # ["aapot/bge-m3-onnx", "https://bge-m3-onnx7-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx7/infer", 1024]
        ],
        "tei_endpoints": [
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://62.146.169.111:8080/embed-medium", 32768],
            ["thenlper/gte-small", "http://62.146.169.111:8080/embed-tiny", 512],
            ["Alibaba-NLP/gte-large-en-v1.5", "http://62.146.169.111:8081/embed-small", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://62.146.169.111:8081/embed-medium", 32768],
            ["thenlper/gte-small", "http://62.146.169.111:8081/embed-tiny", 512],
            ["Alibaba-NLP/gte-large-en-v1.5", "http://62.146.169.111:8082/embed-small", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://62.146.169.111:8082/embed-medium", 32768],
            ["thenlper/gte-small", "http://62.146.169.111:8082/embed-tiny", 512],
            ["Alibaba-NLP/gte-large-en-v1.5", "http://62.146.169.111:8083/embed-small", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://62.146.169.111:8083/embed-medium", 32768],
            ["thenlper/gte-small", "http://62.146.169.111:8083/embed-tiny", 512]
        ]
    }
    ipfs_accelerate_py = ipfs_accelerate_py(resources, metadata)
    asyncio.run(ipfs_kit.__test__(resources, metadata))
    print("test complete")
