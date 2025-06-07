import pyarrow.parquet as pq
import pyarrow as pa
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import time
from tqdm.autonotebook import tqdm
from concurrent.futures import ThreadPoolExecutor

input_files = [
    "/ammar_storage/bnwiki_all_abstract.parquet",
    "/ammar_storage/ugwiki_all_abstract.parquet",
]
output_files = [
    "/ammar_storage/output_bnwiki.parquet",
    "/ammar_storage/output_ugwiki.parquet",
]
temp_folders = [
    "/ammar_storage/temp-gpu0/",
    "/ammar_storage/temp-gpu1/",

]

for folder in temp_folders:
    os.makedirs(folder, exist_ok=True)

def process_and_save_row(model, row, row_index, temp_folder):
    sentence = row['Abstract']
    version_control = row['Version Control']
    
    if pd.isna(sentence) or sentence.strip() == '':
        return

    embedding = model.encode(sentence)
    embedding_list = list(embedding)

    wiki_language_value = "bnwiki"
    wiki_language_array = pa.array([wiki_language_value])
    embedding_array = pa.array([embedding_list], type=pa.list_(pa.float32()))
    version_control_array = pa.array([version_control])

    schema = pa.schema([
        pa.field('Wiki Language', pa.string()),
        pa.field('embeddings', pa.list_(pa.float32())),
        pa.field('Version Control', pa.string())
    ])
    table = pa.table({
        'Wiki Language': wiki_language_array,
        'embeddings': embedding_array,
        'Version Control': version_control_array
    }, schema=schema)

    temp_file = os.path.join(temp_folder, f'row_{row_index}.parquet')
    pq.write_table(table, temp_file)

def combine_parquet_files(temp_folder, output_file):
    temp_files = [os.path.join(temp_folder, f) for f in os.listdir(temp_folder) if f.endswith('.parquet')]
    tables = [pq.read_table(file) for file in temp_files]
    combined_table = pa.concat_tables(tables)
    pq.write_table(combined_table, output_file)

def process_file(input_file, output_file, temp_folder, gpu_device):
    model = SentenceTransformer("BAAI/bge-m3", device=gpu_device)
    table = pq.read_table(input_file)
    df = table.to_pandas()

    total_rows = len(df)
    
    with tqdm(total=total_rows, desc=f"Processing {input_file} on {gpu_device}") as pbar:
        for row_index, row in df.iterrows():
            process_and_save_row(model, row, row_index, temp_folder)
            pbar.update(1)

    combine_parquet_files(temp_folder, output_file)
    
    for file in os.listdir(temp_folder):
        os.remove(os.path.join(temp_folder, file))

def main():
    start_time = time.time()

    num_files = len(input_files)
    num_gpus = len(temp_folders)

    if num_files > num_gpus:
        raise ValueError("Number of input files exceeds the number of GPUs available.")

    with ThreadPoolExecutor(max_workers=num_files) as executor:
        futures = [
            executor.submit(process_file, input_files[i], output_files[i], temp_folders[i], f"cuda:{i}")
            for i in range(num_files)
        ]
        for future in futures:
            future.result()

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Operation took {elapsed_time:.2f} minutes")

if __name__ == "__main__":
    main()
