import pyarrow as pa
import pyarrow.parquet as pq
import requests
import os
import time
from pathlib import Path
import sys

def create_embeddings(input_text, server_url):
    response = requests.post(server_url, json={"inputs": input_text}, headers={'Content-Type': 'application/json'})
    embedding = response.json()

    if isinstance(embedding, list):
        return embedding
    raise ValueError("Unexpected API response format")

def process_parquet_file(input_file, output_file, process_all, checkpoint_interval, checkpoint_folder, wiki_language, use_checkpoints, server_url):
    table = pq.read_table(input_file)
    rows_to_process = min(table.num_rows, 50) if not process_all else table.num_rows

    if use_checkpoints:
        os.makedirs(checkpoint_folder, exist_ok=True)

    all_embeddings = []
    all_version_control = []
    all_wiki_language = []

    for i in range(rows_to_process):
        abstract = table.column('Abstract')[i].as_py()
        version_control = table.column('Version Control')[i].as_py()

        if not abstract:
            print(f"Skipping row {i + 1} due to empty abstract")
            continue

        embeddings = create_embeddings(abstract, server_url)
        all_embeddings.append(embeddings)
        all_version_control.append(version_control)
        all_wiki_language.append(wiki_language)

        print(f"Processing {i + 1}th row")

        if use_checkpoints and (i + 1) % checkpoint_interval == 0:
            checkpoint_table = pa.Table.from_pydict({
                'Wiki Language': pa.array(all_wiki_language),
                'Embeddings': pa.array(all_embeddings),
                'Version Control': pa.array(all_version_control)
            })
            checkpoint_file = os.path.join(checkpoint_folder, f'checkpoint_{i + 1}.parquet')
            pq.write_table(checkpoint_table, checkpoint_file)
            print(f"Checkpoint saved at row {i + 1}")
            all_embeddings.clear()
            all_version_control.clear()
            all_wiki_language.clear()

    if all_embeddings:
        final_table = pa.Table.from_pydict({
            'Wiki Language': pa.array(all_wiki_language),
            'Embeddings': pa.array(all_embeddings),
            'Version Control': pa.array(all_version_control)
        })
        pq.write_table(final_table, output_file)
    else:
        combined_tables = []
        if use_checkpoints:
            for file in sorted(os.listdir(checkpoint_folder)):
                if file.endswith(".parquet"):
                    combined_tables.append(pq.read_table(os.path.join(checkpoint_folder, file)))
        if combined_tables:
            combined_table = pa.concat_tables(combined_tables)
            pq.write_table(combined_table, output_file)

    print(f"Final embeddings saved to {output_file}")

if __name__ == "__main__":
    input_file = input("Enter the input file name (Parquet format): ").strip()
    output_file = input("Enter the output file name (Parquet format): ").strip()

    if not output_file.lower().endswith('.parquet'):
        print("Error: Output file must be a Parquet file (.parquet)")
        sys.exit(1)

    input_file = Path(input_file).resolve()
    output_file = Path(output_file).resolve()

    if not output_file.exists():
        with open(output_file, 'wb') as f:
            pass

    wiki_language = input("Enter the Wiki Language (e.g., enwiki, dewiki): ").strip()
    server_url = input("Enter the embedding server URL (e.g., http://159.69.148.218:80/embed): ").strip()

    process_all = input("Do you want to process the entire file? (yes/no): ").strip().lower() == 'yes'
    use_checkpoints = input("Do you want to use checkpoints? (yes/no): ").strip().lower() == 'yes'
    checkpoint_interval = int(input("Please specify the checkpoint interval (number of rows): ").strip()) if use_checkpoints else None
    checkpoint_folder = input("Please specify the checkpoint folder name: ").strip() if use_checkpoints else None

    start_time = time.time()

    process_parquet_file(input_file, output_file, process_all, checkpoint_interval, checkpoint_folder, wiki_language, use_checkpoints, server_url)

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Processing completed in {elapsed_time:.2f} minutes")
