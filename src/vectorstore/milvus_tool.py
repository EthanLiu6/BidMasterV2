from pymilvus import MilvusClient
from typing import Dict, List, Union


def insert_data(
        milvus_client_object: MilvusClient,
        collection_name: str,
        data: Union[Dict, List[Dict]],
) -> None:
    """
    Args:
        milvus_client_object: MilvusClient对象, 防止连续insert的时候连续创建
        collection_name: pass
        data: pass. Note: 一定要和collection纬度一致（dim一致）
        init_store: 是初始化存储还是插入新数据
    """
    client = milvus_client_object
    collections = client.list_collections()
    print("当前数据库的collection:\n", collections)
    if collection_name in collections:
        print("当前待插入collection信息:\n", client.describe_collection(collection_name))

        # 获取 Collection 统计信息（包含数据量）
        stats = client.get_collection_stats(collection_name=collection_name)
        num_entities = stats["row_count"]
        print(f"插入前Collection 数据数量: {num_entities}")
        client.insert(collection_name=collection_name, data=data)

    else:
        raise 'collection名不在当前数据库中'
