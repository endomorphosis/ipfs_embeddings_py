from accelerate import Accelerator, PartialState
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
import time

start_time = time.time()

accelerator = Accelerator()
distributed_state = PartialState()

print("Loading data...")
df = pd.read_parquet('bnwiki_all_abstract.parquet')
df = df.head(1000)

print("Initializing model...")
model_name = 'BAAI/bge-m3'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model = model.to(accelerator.device)
model = accelerator.prepare(model)

def compute_embeddings(texts, tokenizer, model):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    inputs = {key: value.to(accelerator.device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

batch_size = 32
embeddings = []
version_control = df['Version Control'].tolist()

print("Processing batches...")
for i in range(0, len(df), batch_size):
    batch_texts = df['Abstract'].iloc[i:i + batch_size].tolist()
    with distributed_state.split_between_processes(batch_texts) as split_batch_texts:
        batch_embeddings = compute_embeddings(split_batch_texts, tokenizer, model)
        embeddings.append(batch_embeddings)

print("Combining embeddings...")
all_embeddings = torch.cat(embeddings).cpu().numpy()

print("Saving results...")
output_df = pd.DataFrame({
    'Embeddings': list(all_embeddings),
    'Version Control': version_control
})
output_df.to_parquet('hf_embed.parquet', index=False)

end_time = time.time()
elapsed_time = (end_time - start_time) / 60

print(f"Processing completed in {elapsed_time:.2f} minutes.")
