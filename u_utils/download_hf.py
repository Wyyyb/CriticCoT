from huggingface_hub import hf_hub_download
import os


def download_files():
    # 设置仓库信息
    repo_id = "ubowang/on_policy_data_0119"

    # 要下载的文件列表
    files = [
        "qwen_math_ace_80k_0119.json",
        "qwen_math_numina_80k_0119.json"
    ]

    # 创建保存目录
    os.makedirs("downloaded_files", exist_ok=True)

    # 下载每个文件
    for filename in files:
        try:
            # 使用 hf_hub_download 下载文件
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir="downloaded_files"
            )
            print(f"Successfully downloaded {filename} to {downloaded_path}")
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")


if __name__ == "__main__":
    # 首先安装必要的包
    # pip install huggingface_hub

    download_files()
