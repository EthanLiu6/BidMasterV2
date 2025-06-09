from utils.common_utils import find_project_root

ROOT_PATH = find_project_root()
MODELS_PATH = find_project_root().joinpath('models')

# 用户问题的初步意图分析，分类问题并将结果进行后续任务（可与chat model相同）
QUERY_ANALYSIS_MODEL_NAME = 'Qwen/Qwen3-1.8B-Chat'
QUERY_ANALYSIS_MODEL_PATH = MODELS_PATH.joinpath(QUERY_ANALYSIS_MODEL_NAME)

# 文本（句子）向量化模型，用于向量数据库存储和Query向量化
SENTENCE_EMBEDDING_MODEL_NAME = 'thenlper/gte-large-zh'
SENTENCE_EMBEDDING_MODEL_PATH = MODELS_PATH.joinpath(SENTENCE_EMBEDDING_MODEL_NAME)

# 用于跟用户进行交流的模型
CHAT_MODEL_NAME = 'Qwen/Qwen3-8B-Chat'
CHAT_MODEL_PTAH = MODELS_PATH.joinpath(CHAT_MODEL_NAME)

# 对召回数据进行重排的模型（暂时不做）
RERANK_MODEL_NAME = None
RERANK_MODEL_PATH = None
