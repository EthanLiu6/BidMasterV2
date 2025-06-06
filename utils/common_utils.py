import csv
import json
import os
from pathlib import Path

import pandas as pd


def find_project_root():
    # 当前脚本路径
    current_file = Path(__file__).resolve()

    # 项目根目录（BidMaster-with-LC 是根）
    project_root = current_file.parents[1]  # 逐级向上：utils -> BidMaster-with-LC
    return project_root


# 默认环境变量获取函数
def get_env(key: str, default=None) -> str:
    return os.getenv(key, default)


def csv2json(csv_file_path, json_file_path=None, encoding='utf-8'):
    data = []
    with open(csv_file_path, encoding=encoding) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # 处理空值，将空字符串转换为None
            processed_row = {k: (v if v != '' else None) for k, v in row.items()}
            data.append(processed_row)

    # 写入JSON文件
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

    print(f"✅ 成功将 CSV 文件 '{csv_file_path}' 转换为 JSON 文件 '{json_file_path}'")


def json2csv(json_file_path, csv_file_path):
    """
    将指定的 JSON 文件转换为 CSV 文件。

    参数：
        json_file_path (str): JSON 文件的路径
        csv_file_path (str): 输出的 CSV 文件路径
    """
    try:
        # 读取 JSON 数据
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise ValueError("JSON 数据格式不符合要求：应为字典对象列表")

        # 写入 CSV 文件
        with open(csv_file_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
            fieldnames = data[0].keys()  # 自动提取字段名（表头）
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(data)

        print(f"✅ 成功将 JSON 文件 '{json_file_path}' 转换为 CSV 文件 '{csv_file_path}'")

    except Exception as e:
        print(f"❌ 转换过程中发生错误：{e}")


def excel2json(origin_file, target_file, origin_sheet_name=0):
    """
    save excel文件为json
    Args:
    origin_file: 原始excel文件
    target_file: 目标json文件
    """

    # 读取 Excel，同时自动解析日期类型
    df = pd.read_excel(origin_file, sheet_name=origin_sheet_name, parse_dates=True)

    # 确保日期类型被转换为 ISO 格式字符串（如：2024-05-21T15:30:00）
    def convert_value(val):
        if pd.isna(val):
            return None
        elif isinstance(val, pd.Timestamp):
            return val.isoformat()
        elif isinstance(val, pd.Timedelta):
            return str(val)
        elif isinstance(val, (int, float, bool, str)):
            return val
        else:
            return str(val)

    # 应用转换
    json_ready = df.map(convert_value).to_dict(orient="records")

    # 保存为 JSON 文件
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(json_ready, f, ensure_ascii=False, indent=4)

    print('succeed!')


def get_dir_files_name(dir_path) -> list[str]:
    """获取指定文件夹下的所有文件名，返回所有文件名的列表格式"""

    if not os.path.isdir(dir_path):
        return []

    files = []
    for entry in os.listdir(dir_path):
        full_path = os.path.join(dir_path, entry)
        if os.path.isfile(full_path):
            files.append(entry)

    return files


if __name__ == '__main__':
    project_path = find_project_root()
    laws_files = get_dir_files_name(project_path.joinpath('processed_data/laws'))
    print(laws_files)
