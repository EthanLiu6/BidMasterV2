from pymilvus import MilvusClient
from src.model_backend.sentence_model import SentenceModel


def test01_dense_law_search(law_query: str):
    client = MilvusClient('../src/vectorstore/db/Bid.db')

    stc_model = SentenceModel('../models/thenlper/gte-large-zh')
    emb, _ = stc_model.encode(law_query)

    res = client.search(
        collection_name='laws_collection',
        data=emb.tolist(),
        anns_field='law_vector',
        output_fields=['from_file', 'law_text']
    )

    for num, each_res in enumerate(res[0]):
        print(f'第{num}条：')
        print(f'distance:{each_res["distance"]}')
        print(f'from_file:{each_res["entity"]["from_file"]}')
        print(f'law_text:{each_res["entity"]["law_text"]}')
        print('*' * 30)


def test02_dense_biaodewu_search(law_query: str):
    client = MilvusClient('../src/vectorstore/db/Bid.db')

    stc_model = SentenceModel('../models/thenlper/gte-large-zh')
    emb, _ = stc_model.encode(law_query)

    res = client.search(
        collection_name='biaodewu_info_collection',
        data=emb.tolist(),
        anns_field='biaodewu_vector',
        output_fields=['biaodewu_info_text']
    )

    for num, each_res in enumerate(res[0]):
        print(f'第{num}条：')
        print(f'distance:{each_res["distance"]}')
        print(f'biaodewu_info_text:{each_res["entity"]["biaodewu_info_text"]}')
        print('*' * 30)


if __name__ == '__main__':
    # query = '开标现场由采购代理机构主持，采购人可以不参与吗？'
    # test01_dense_law_search(query)

    query = '最近有哪些企业采购的中标信息？'
    test02_dense_biaodewu_search(query)
