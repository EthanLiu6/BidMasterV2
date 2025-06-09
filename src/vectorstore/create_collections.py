from pymilvus import MilvusClient, DataType

__all__ = [
    'create_laws_collection',
    'create_biaodewu_info_collection'
]


def create_laws_collection(
        milvus_server_address: str,
        encode_dim: int,
        collection_name: str = 'laws_collection',
        **kwargs
) -> None:
    """
    Args:
        milvus_server_address: milvus数据库位置或者服务器地址
        encode_dim: 句子向量化纬度（sentence embedding dim）
        collection_name: 创建数据库集合（表）名，因为是法律数据，默认使用'laws_collection'
        kwargs: 其他的一些可能参数，比如数据库的密码、Token等。如果有，需要修改MilvusClient实例化代码

    Details: 创建的集合（表）中有几个field（字段）: law_id、law_vector、law_text、from_file
    """
    # TODO: set 'collection_name' to 'config'

    client = MilvusClient(
        uri=str(milvus_server_address)
    )

    client.drop_collection(collection_name)
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)

    schema.add_field(field_name="law_id", datatype=DataType.INT64, is_primary=True,
                     description="every law id")
    schema.add_field(field_name="law_vector", datatype=DataType.FLOAT_VECTOR, dim=encode_dim,
                     description="every law vector")
    schema.add_field(field_name="law_text", datatype=DataType.VARCHAR, max_length=65535,
                     description="every law text")
    schema.add_field(field_name="from_file", datatype=DataType.VARCHAR, max_length=1024,
                     description="which file each law is located")

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="law_vector", index_type="IVF_FLAT", metric_type="IP")
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    print(f'{collection_name}创建成功！')
    client.close()


def create_biaodewu_info_collection(
        milvus_server_address: str,
        encode_dim: int,
        collection_name: str = 'biaodewu_info_collection',
        **kwargs
) -> None:
    """
    Args:
        milvus_server_address: milvus数据库位置或者服务器地址
        encode_dim: 句子向量化纬度（sentence embedding dim）
        collection_name: 创建数据库集合（表）名，因为是标的物信息数据，默认使用'biaodewu_info_collection'
        kwargs: 其他的一些可能参数，比如数据库的密码、Token等。如果有，需要修改MilvusClient实例化代码

    Details: 创建的集合（表）中有几个field（字段）: …………
    """

    client = MilvusClient(
        uri=str(milvus_server_address)
    )
    client.drop_collection(collection_name)
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)

    schema.add_field(field_name="bid_info_id", datatype=DataType.INT64, is_primary=True, auto_id=True,
                     description="biaodewu info id")
    schema.add_field(field_name="biaodewu_vector", datatype=DataType.FLOAT_VECTOR, dim=encode_dim,
                     description="biaodewu info vector")
    schema.add_field(field_name="biaodewu_info_text", datatype=DataType.VARCHAR, max_length=65535,
                     description="biaodewu info text")
    schema.add_field(field_name="bid_type", datatype=DataType.VARCHAR, max_length=32,
                     description="the type of bid, like: Robot")

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="biaodewu_vector", index_type="IVF_FLAT", metric_type="IP")
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    print(f'{collection_name}创建成功！')
    client.close()


if __name__ == '__main__':
    milvus_db = './db/Bid.db'
    create_laws_collection(
        milvus_server_address=milvus_db,
        encode_dim=1024
    )

    create_biaodewu_info_collection(
        milvus_server_address=milvus_db,
        encode_dim=1024
    )
