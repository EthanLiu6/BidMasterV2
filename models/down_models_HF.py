from huggingface_hub import snapshot_download


def download_model(_model_name: str, _local_dir: str = "./xxx"):
    snapshot_download(repo_id=_model_name, local_dir=_local_dir)


if __name__ == "__main__":
    model_name = "BAAI/bge-large-zh-v1.5"
    local_dir = "./BAAI/bge-large-zh-v1.5"
    download_model(model_name, local_dir)
    print(f"Model {model_name} downloaded to {local_dir}")