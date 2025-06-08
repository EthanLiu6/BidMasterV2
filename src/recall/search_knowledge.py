from pymilvus import MilvusClient


def dense_law_search(
        milvus_client_object: MilvusClient,
        query_emb: list[list | float],
        collection_name: str = 'laws_collection',
        limit: int = 8
):
    """全文密集搜索相关法律信息"""

    client = milvus_client_object

    res = client.search(
        collection_name=collection_name,
        data=query_emb,
        anns_field='law_vector',
        output_fields=['law_vector', 'law_text', 'from_file'],
        limit=limit
    )

    return res


def dense_biaodewu_search(
        milvus_client_object: MilvusClient,
        query_emb: list[list | float],
        collection_name: str = 'biaodewu_info_collection',
        limit: int = 8
):
    """全文密集搜索相关标的物信息"""

    client = milvus_client_object

    res = client.search(
        collection_name=collection_name,
        data=query_emb,
        anns_field='biaodewu_vector',
        output_fields=['biaodewu_vector', 'biaodewu_info_text', 'bid_type'],
        limit=limit
    )

    return res


def sparse_law_search():
    """稀疏搜索相关法律信息"""
    pass


def sparse_biaodewu_search():
    """稀疏搜索相关标的物信息"""
    pass


def hybrid_law_search():
    """混合搜索相关法律信息"""
    pass


def hybrid_biaodewu_search():
    """混合搜索相关标的物信息"""
    pass


if __name__ == '__main__':
    from src.model_backend.sentence_model import SentenceModel

    _client = MilvusClient('../vectorstore/db/Bid.db')
    stc_model = SentenceModel('../../models/thenlper/gte-large-zh')

    # query = '开标现场由采购代理机构主持，采购人可以不参与吗？'
    # query = '把赠予的款项写入招标文件可以吗？'
    query = '偷拍的照片，可以作为投诉“证据”吗？'
    emb, _ = stc_model.encode(query)
    _res = dense_law_search(_client, emb.tolist())
    res_law_texts = []
    distances = []
    for num, each_res in enumerate(_res[0]):
        print(f'第{num}条：')
        print(f'distance:{each_res["distance"]}')
        print(f'law_vector:{each_res["entity"]["law_vector"]}')
        print(f'law_text:{each_res["entity"]["law_text"]}')
        print(f'from_file:{each_res["entity"]["from_file"]}')
        print('*' * 30)
        res_law_texts.append(each_res["entity"]["law_text"])
        distances.append(each_res["distance"])
    print(res_law_texts)
    print(distances)  # [0.729775071144104, 0.7288713455200195, 0.723609209060669, 0.7187605500221252, 0.712833821773529, 0.7097634673118591, 0.7072374224662781, 0.7041965126991272]


    # query = '最近有哪些企业采购的中标信息？'
    # emb, _ = stc_model.encode(query)
    # _res = dense_biaodewu_search(_client, emb.tolist())
    # for num, each_res in enumerate(_res[0]):
    #     print(f'第{num}条：')
    #     print(f'distance:{each_res["distance"]}')
    #     print(f'biaodewu_vector:{each_res["entity"]["biaodewu_vector"]}')
    #     print(f'biaodewu_info_text:{each_res["entity"]["biaodewu_info_text"]}')
    #     print(f'bid_tpye:{each_res["entity"]["bid_type"]}')
    #     print('*' * 30)
