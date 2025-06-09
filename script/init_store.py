from src.vectorstore.vector_store import *

from config.data_config import *
from config.model_config import SENTENCE_EMBEDDING_MODEL
from config.milvus_db_config import *

__all__ = [
    'init_store_laws_and_bid'
]


def init_store_laws_and_bid():
    print("初始化存储法律和标的物数据中………")

    store_init_laws(
        laws_dir_path=LAWS_DIR_PATH,
        sentence_model=SENTENCE_EMBEDDING_MODEL,
        encode_dim=SENTENCE_EMBEDDING_MODEL.dim,
        milvus_server_address=str(DB_PATH),
        batch_size=6
    )

    store_init_biaodewu_infos(
        biaodewu_info_dir_path=BIAODEWU_DIR_PATH,
        sentence_model=SENTENCE_EMBEDDING_MODEL,
        encode_dim=SENTENCE_EMBEDDING_MODEL.dim,
        milvus_server_address=str(DB_PATH),
        batch_size=8
    )


if __name__ == '__main__':
    init_store_laws_and_bid()
