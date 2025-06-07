from autofaiss import build_index

build_index(
    embeddings="/mnt/nvme/ammar_faiss/german",
    index_path="knn.index",
    index_infos_path="infos.json",
    current_memory_available="90GB",
    use_gpu=True,
    should_be_memory_mappable=True,
    verbose=20,
    max_index_memory_usage="4GB",
    nb_cores=48,
    metric_type="l2",
    save_on_disk="/mnt/nvme/ammar_faiss/auto_german"
)
