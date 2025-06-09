import json
from pathlib import Path

from pymilvus import MilvusClient

from src.vectorstore.milvus_tool import insert_data, upsert_data
from utils import common_utils
from src.model_backend.sentence_model import SentenceModel

__all__ = [
    'store_init_laws',
    'store_init_biaodewu_infos',
    'store_add_law',
    'store_add_biaodewu_info'
]


def store_init_laws(
        laws_dir_path: Path,
        sentence_model: SentenceModel,
        encode_dim: int,
        milvus_server_address,
        collection_name: str = 'laws_collection',
        batch_size=8
):
    """是对给定文件夹下的json格式的法律数据进行存储(初始化存储)"""

    # init store的时候确保collection内无数据（新建）
    from src.vectorstore.create_collections import create_laws_collection
    create_laws_collection(
        milvus_server_address=milvus_server_address,
        encode_dim=encode_dim,
        collection_name=collection_name
    )

    milvus_server_address = str(milvus_server_address)
    client = MilvusClient(
        uri=milvus_server_address
    )

    laws_files = common_utils.get_dir_files_name(laws_dir_path)
    for cur_law_file_name in laws_files:
        if cur_law_file_name.endswith('.json'):
            cur_law_name = cur_law_file_name.split('.json')[0]
            with open(laws_dir_path.joinpath(cur_law_file_name), 'r') as law_file:
                cur_law_data = json.load(law_file)

            # Note: 与laws的json数据格式一致
            law_texts = [law['条款'] + '\n' + law['内容'] for law in cur_law_data]
            print("当前文档内容条数：", len(law_texts))

            # Note: 与create_laws_collection字段保持一致
            # 防止内存溢出，按批次进行
            for batch_idx in range(0, len(law_texts), batch_size):
                if batch_idx >= len(law_texts):
                    batch_idx = len(law_texts) - 1
                batch_law_texts = law_texts[batch_idx: batch_idx + batch_size]
                batch_law_texts_emb, batch_emb_shape = sentence_model.encode(batch_law_texts)
                print("当前batch文档内容向量化后的shape：", batch_emb_shape)

                batch_data = [
                    {
                        "law_vector": batch_law_text_emb.tolist(),
                        "law_text": batch_law_text,
                        "from_file": cur_law_name
                    }
                    for batch_law_text_emb, batch_law_text in zip(batch_law_texts_emb, batch_law_texts)
                ]
                insert_data(
                    milvus_client_object=client,
                    collection_name=collection_name,
                    data=batch_data,
                )

            print(f"current {cur_law_file_name} was inserted！")

    client.close()


def store_add_law(
        laws_file_path: Path,
        sentence_model: SentenceModel,
        milvus_server_address,
        collection_name: str = 'laws_collection',
        batch_size=8
):
    """是对给定文件夹下的json格式的法律数据进行存储(新增存储), dim要一致"""

    milvus_server_address = str(milvus_server_address)
    client = MilvusClient(
        uri=milvus_server_address
    )

    collections = client.list_collections()
    if collection_name not in collections:
        print("当前数据库包含的collection：", collections)
        raise '当前数据库中无该collection'

    cur_law_file_name = common_utils.get_path_file_name(laws_file_path)

    if cur_law_file_name.endswith('.json'):
        cur_law_name = cur_law_file_name.split('.json')[0]
        with open(laws_file_path, 'r') as law_file:
            cur_law_data = json.load(law_file)

        # Note: 与laws的json数据格式一致
        law_texts = [law['条款'] + '\n' + law['内容'] for law in cur_law_data]
        print("当前文档内容条数：", len(law_texts))

        # Note: 与create_laws_collection字段保持一致
        # 防止内存溢出，按批次进行
        for batch_idx in range(0, len(law_texts), batch_size):
            if batch_idx >= len(law_texts):
                batch_idx = len(law_texts) - 1
            batch_law_texts = law_texts[batch_idx: batch_idx + batch_size]
            batch_law_texts_emb, batch_emb_shape = sentence_model.encode(batch_law_texts)
            print("当前batch文档内容向量化后的shape：", batch_emb_shape)

            batch_data = [
                {
                    "law_vector": batch_law_text_emb.tolist(),
                    "law_text": batch_law_text,
                    "from_file": cur_law_name
                }
                for batch_law_text_emb, batch_law_text in zip(batch_law_texts_emb, batch_law_texts)
            ]
            upsert_data(
                milvus_client_object=client,
                collection_name=collection_name,
                data=batch_data,
            )

        print(f"current {cur_law_file_name} was upsert！")

    else:
        raise '需要满足json格式的条文数据'

    client.close()


def store_init_biaodewu_infos(
        biaodewu_info_dir_path: Path,
        sentence_model: SentenceModel,
        encode_dim: int,
        milvus_server_address,
        collection_name: str = 'biaodewu_info_collection',
        batch_size=8
):
    """是对给定文件夹下的json格式的标的物数据进行存储(初始化存储)"""

    # init store的时候确保collection内无数据（新建）
    from src.vectorstore.create_collections import create_biaodewu_info_collection
    create_biaodewu_info_collection(
        milvus_server_address=milvus_server_address,
        encode_dim=encode_dim,
        collection_name=collection_name
    )

    milvus_server_address = str(milvus_server_address)
    client = MilvusClient(
        uri=milvus_server_address
    )

    biaodewu_info_files = common_utils.get_dir_files_name(biaodewu_info_dir_path)
    for cur_biaodewu_info_file_name in biaodewu_info_files:
        if cur_biaodewu_info_file_name.endswith('.json'):
            cur_biaodewu_info_name = cur_biaodewu_info_file_name.split('.json')[0]
            with open(biaodewu_info_dir_path.joinpath(cur_biaodewu_info_file_name), 'r') as biaodewu_info_file:
                cur_biaodewu_info_data = json.load(biaodewu_info_file)

            # Note: 与biaodewu_info的json数据格式一致
            biaodewu_info_texts = [str(biaodewu_info) for biaodewu_info in cur_biaodewu_info_data]
            print("当前文档内容条数：", len(biaodewu_info_texts))

            # Note: 与create_laws_collection字段保持一致
            # 防止内存溢出，按批次进行
            for batch_idx in range(0, len(biaodewu_info_texts), batch_size):
                if batch_idx >= len(biaodewu_info_texts):
                    batch_idx = len(biaodewu_info_texts) - 1
                batch_biaodewu_info_texts = biaodewu_info_texts[batch_idx: batch_idx + batch_size]
                batch_biaodewu_info_texts_emb, batch_emb_shape = sentence_model.encode(batch_biaodewu_info_texts)
                print("当前batch文档内容向量化后的shape：", batch_emb_shape)

                batch_data = [
                    {
                        "biaodewu_vector": batch_biaodewu_info_text_emb.tolist(),
                        "biaodewu_info_text": batch_biaodewu_info_text,
                        "bid_type": cur_biaodewu_info_name
                    }
                    for batch_biaodewu_info_text_emb, batch_biaodewu_info_text in
                    zip(batch_biaodewu_info_texts_emb, batch_biaodewu_info_texts)
                ]
                insert_data(
                    milvus_client_object=client,
                    collection_name=collection_name,
                    data=batch_data,
                )

            print(f"current {cur_biaodewu_info_file_name} was inserted！")

    client.close()


def store_add_biaodewu_info():
    pass


if __name__ == '__main__':
    project_path = common_utils.find_project_root()
    stc_model_path = project_path.joinpath('models/thenlper/gte-large-zh')
    stc_model = SentenceModel(model_path=stc_model_path)
    print("使用模型的维度：", stc_model.dim)
    print("使用模型的device：", stc_model.device)

    laws_dir = project_path.joinpath('processed_data/laws')
    db_path = str(project_path.joinpath('src/vectorstore/db/Bid.db'))
    store_init_laws(laws_dir,
                    sentence_model=stc_model,
                    encode_dim=stc_model.dim,
                    milvus_server_address=db_path,
                    collection_name='laws_collection')

    # from create_collections import create_laws_collection
    # test_db_path = str(project_path.joinpath('test/test.db'))
    # create_laws_collection(test_db_path, stc_model.dim, 'test_collection')
    # law_file_path = project_path.joinpath('processed_data/laws/招标投标领域公平竞争审查规则.json')
    # store_add_law(
    #     laws_file_path=law_file_path,
    #     sentence_model=stc_model,
    #     milvus_server_address=test_db_path,
    #     collection_name='test_collection'
    # )

    biaodewu_info_dir = project_path.joinpath('processed_data/biaodewu')
    db_path = str(project_path.joinpath('src/vectorstore/db/Bid.db'))
    store_init_biaodewu_infos(biaodewu_info_dir,
                              sentence_model=stc_model,
                              encode_dim=stc_model.dim,
                              milvus_server_address=db_path,
                              collection_name='biaodewu_info_collection'
                              )
