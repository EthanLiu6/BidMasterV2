import numpy as np
from pymilvus import MilvusClient, DataType


def test01_create_collection():
    entities = [
        {'doc_id': 0, 'doc_vector': np.array([-0.0372721, 0.0101959, -0.114994]),
         'doc_text': "In 1950, Alan Turing published his seminal paper, 'Computing Machinery and Intelligence,' proposing the Turing Test as a criterion of intelligence, a foundational concept in the philosophy and development of artificial intelligence."},
        {'doc_id': 1, 'doc_vector': np.array([-0.00308882, -0.0219905, -0.00795811]),
         'doc_text': "The Dartmouth Conference in 1956 is considered the birthplace of artificial intelligence as a field; here, John McCarthy and others coined the term 'artificial intelligence' and laid out its basic goals."},
        {'doc_id': 2, 'doc_vector': np.array([0.00945078, 0.00397605, -0.0286199], dtype=np.float32),
         'doc_text': 'In 1951, British mathematician and computer scientist Alan Turing also developed the first program designed to play chess, demonstrating an early example of AI in game strategy.'},
        {'doc_id': 3, 'doc_vector': np.array([-0.0391119, -0.00880096, -0.0109257], dtype=np.float32),
         'doc_text': 'The invention of the Logic Theorist by Allen Newell, Herbert A. Simon, and Cliff Shaw in 1955 marked the creation of the first true AI program, which was capable of solving logic problems, akin to proving mathematical theorems.'}
    ]

    client = MilvusClient(
        uri="./test.db"  # replace with your own Milvus server address
    )

    client.drop_collection('test_collection')

    schema = client.create_schema(auto_id=False, enabel_dynamic_field=True)

    schema.add_field(field_name="doc_id", datatype=DataType.INT64, is_primary=True, description="document id")
    schema.add_field(field_name="doc_vector", datatype=DataType.FLOAT_VECTOR, dim=3, description="document vector")
    schema.add_field(field_name="doc_text", datatype=DataType.VARCHAR, max_length=65535, description="document text")

    index_params = client.prepare_index_params()

    index_params.add_index(field_name="doc_vector", index_type="IVF_FLAT", metric_type="IP", params={"nlist": 128})

    client.create_collection(collection_name="test_collection", schema=schema, index_params=index_params)

    client.insert(collection_name="test_collection", data=entities)


def test02_search_data():
    client = MilvusClient(
        uri="./test.db"  # replace with your own Milvus server address
    )
    res = client.search(
        collection_name="test_collection",
        data=[[-0.045217834, 0.035171617, -0.025117004]],  # replace with your query vector
        limit=3,
        output_fields=["doc_id", "doc_text"]
    )

    for i in res[0]:
        print(f'distance: {i["distance"]}')
        print(f'doc_text: {i["entity"]["doc_text"]}')


def test03_rerank_data():
    from pymilvus.model.reranker import CrossEncoderRerankFunction

    ce_rf = CrossEncoderRerankFunction(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",  # Specify the model name.
        device="cpu"  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    )

    reranked_results = ce_rf(
        query='What event in 1956 marked the official birth of artificial intelligence as a discipline?',
        documents=[
            "In 1950, Alan Turing published his seminal paper, 'Computing Machinery and Intelligence,' proposing the Turing Test as a criterion of intelligence, a foundational concept in the philosophy and development of artificial intelligence.",
            "The Dartmouth Conference in 1956 is considered the birthplace of artificial intelligence as a field; here, John McCarthy and others coined the term 'artificial intelligence' and laid out its basic goals.",
            "In 1951, British mathematician and computer scientist Alan Turing also developed the first program designed to play chess, demonstrating an early example of AI in game strategy.",
            "The invention of the Logic Theorist by Allen Newell, Herbert A. Simon, and Cliff Shaw in 1955 marked the creation of the first true AI program, which was capable of solving logic problems, akin to proving mathematical theorems."
        ],
        top_k=3
    )

    for result in reranked_results:
        print(f'score: {result.score}')
        print(f'doc_text: {result.text}')


if __name__ == '__main__':
    # test01_create_collection()
    # test02_search_data()
    test03_rerank_data()
