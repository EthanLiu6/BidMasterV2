from pymilvus import MilvusClient

from src.model_backend import Qwen3Model
from src.model_backend import SentenceModel
from src.recall.search_knowledge import dense_law_search

from src.prompt import LawPrompt


def bid_laws_chat(
        llm_model: Qwen3Model,
        sentence_model: SentenceModel,
        milvus_client: MilvusClient,
        query: str,
        _collection_name: str = 'laws_collection',
        thinking: bool = True
) -> str:
    query_emb, _ = sentence_model.encode(query)
    searched_res = dense_law_search(milvus_client, query_emb.tolist(), _collection_name)
    res_law_texts = [each_res["entity"]["law_text"] for each_res in searched_res[0]]

    law_prompt = LawPrompt().build_prompt(
        question=query,
        context=res_law_texts
    )

    return llm_model.generate(prompt=law_prompt, enable_thinking=thinking)


if __name__ == '__main__':
    from utils.common_utils import find_project_root
    root_path = find_project_root()

    qwen3 = Qwen3Model(model_path=root_path.joinpath('models/Qwen/Qwen3-8B'))
    stc_model = SentenceModel(model_path=root_path.joinpath('models/thenlper/gte-large-zh'))
    client = MilvusClient(str(root_path.joinpath('src/vectorstore/db/Bid.db')))
    while 1:
        user_query = input("请输入你咨询的招投标法律相关问题(q退出):\n")
        if user_query == 'q':
            break

        answer = bid_laws_chat(qwen3, stc_model, client, user_query, thinking=False)
        print(answer)
