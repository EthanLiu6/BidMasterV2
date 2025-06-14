from src.model_backend import Qwen3Model

from src.prompt import QueryAnalysisPrompt
from config.data_config import QUERY_CLS


__all__ = ['query_cls_chat']


def query_cls_chat(
        llm_model: Qwen3Model,
        query: str,
        thinking: bool = True
) -> str:
    bid_info_prompt = QueryAnalysisPrompt().build_prompt(
        question=query,
        context=QUERY_CLS
    )

    return llm_model.generate_unstream(prompt=bid_info_prompt, enable_thinking=thinking)


if __name__ == '__main__':
    from utils.common_utils import find_project_root

    root_path = find_project_root()

    qwen3 = Qwen3Model(model_path=root_path.joinpath('models/Qwen/Qwen3-0.6B'))
    while 1:
        user_query = input("请输入你咨询的招投标相关问题(q退出):\n")
        if user_query == 'q':
            break

        answer = query_cls_chat(qwen3, user_query, thinking=False)
        print(answer)
