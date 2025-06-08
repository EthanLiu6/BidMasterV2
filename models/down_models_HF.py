from huggingface_hub import snapshot_download

def download_model(_model_name: str, _local_dir: str = "./xxx"):
    snapshot_download(
        repo_id=_model_name,
        local_dir=_local_dir,
        allow_patterns=["*.safetensors"],  # 仅允许下载 safetensors 文件
        ignore_patterns=["*.bin", "*.h5", "*.pt"],  # 显式忽略其他权重格式
    )

if __name__ == "__main__":
    model_name = "thenlper/gte-large-zh"
    local_dir = "./thenlper/gte-large-zh"
    download_model(model_name, local_dir)
    print(f"Model {model_name} downloaded to {local_dir}")