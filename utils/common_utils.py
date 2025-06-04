import os
from pathlib import Path


def find_project_root():
    # 当前脚本路径
    current_file = Path(__file__).resolve()

    # 项目根目录（BidMaster-with-LC 是根）
    project_root = current_file.parents[1]  # 逐级向上：utils -> BidMaster-with-LC
    return project_root


# 默认环境变量获取函数
def get_env(key: str, default=None) -> str:
    return os.getenv(key, default)
