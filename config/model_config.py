from config import ROOT_PATH
from src.model_backend import SentenceModel, Qwen3Model

__all__ = [
    'CHAT_MODEL',
    'SENTENCE_EMBEDDING_MODEL',
    'QUERY_ANALYSIS_MODEL'
]

# 所有模型存放地址
MODELS_PATH = ROOT_PATH.joinpath('models')

# 用户问题的初步意图分析，分类问题并将结果进行后续任务（可与chat model相同）
QUERY_ANALYSIS_MODEL_NAME = 'Qwen/Qwen3-0.6B'  # 'Qwen/Qwen3-1.8B-Chat'
QUERY_ANALYSIS_MODEL_PATH = MODELS_PATH.joinpath(QUERY_ANALYSIS_MODEL_NAME)
QUERY_ANALYSIS_MODEL = Qwen3Model(
    model_path=str(QUERY_ANALYSIS_MODEL_PATH),
    model_name='Query_analysis_model:' + QUERY_ANALYSIS_MODEL_NAME
)

# 文本（句子）向量化模型，用于向量数据库存储和Query向量化
SENTENCE_EMBEDDING_MODEL_NAME = 'thenlper/gte-large-zh'
SENTENCE_EMBEDDING_MODEL_PATH = MODELS_PATH.joinpath(SENTENCE_EMBEDDING_MODEL_NAME)
SENTENCE_EMBEDDING_MODEL = SentenceModel(
    model_path=str(SENTENCE_EMBEDDING_MODEL_PATH),
    model_name='Embedding_model:' + SENTENCE_EMBEDDING_MODEL_NAME
)

# 用于跟用户进行交流的模型
CHAT_MODEL_NAME = 'Qwen/Qwen3-0.6B'  # 'Qwen/Qwen3-8B-Chat'
CHAT_MODEL_PTAH = MODELS_PATH.joinpath(CHAT_MODEL_NAME)
if CHAT_MODEL_NAME == QUERY_ANALYSIS_MODEL_NAME:
    CHAT_MODEL = QUERY_ANALYSIS_MODEL

else:
    CHAT_MODEL = Qwen3Model(
        model_path=str(CHAT_MODEL_PTAH),
        model_name='Chat_model:' + CHAT_MODEL_NAME
    )

# 对召回数据进行重排的模型（暂时不做）
RERANK_MODEL_NAME = None
RERANK_MODEL_PATH = None
