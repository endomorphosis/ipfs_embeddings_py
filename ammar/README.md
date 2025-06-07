Please, use ```create_gaudi.py``` file to generate embeddings on Intel Gaudi2 server. In the provided code: I am giving a large payload of all the abstract of a single row to the gaudi and getting a list of embeddings and saving it. A bity unorthodox but we chose this method because M3 model can handle large payloads. 

### Comments

Can you use batch?
- A mixed answer. Because the method I use of (large paylods) a batch method is not sustainable since it requires a good amount of workers (threads) to process this at scale. It requires me 10 hours to generate 7M embeddings. If you essentially have multiple HPU (Gaudi node) still by long run it's not feasible unless these HPUs are on different systems altogether. 

How about reverse proxy and using multiple HPUs?
- Answer lies on your system load. In my case, I was handling with huge amounts of data. You can consider each of my payload as 1/1 and half text-book page payload. If we talking about such scale, you can't use one Gaudi server, you need multiple servers so that your system can sustain the load my code will be putting.

- GPU code ios still experimental, if it breaks I can't be blamed.
