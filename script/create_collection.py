from config.milvus_db_config import *
from src.vectorstore.create_collections import *

__all__ = [
    'create_law_and_bid_collection'
]


def create_law_and_bid_collection():
    print("创建法律和标的物信息表结构中……")

    create_laws_collection(
        milvus_server_address=str(DB_PATH),
        encode_dim=SENTENCE_EMBEDDING_MODEL_DIM,
        collection_name=LAWS_COLLECTION_NAME
    )

    create_biaodewu_info_collection(
        milvus_server_address=str(DB_PATH),
        encode_dim=SENTENCE_EMBEDDING_MODEL_DIM,
        collection_name=BIAODEWU_INFO_COLLECTION_NAME
    )


if __name__ == '__main__':
    create_law_and_bid_collection()
