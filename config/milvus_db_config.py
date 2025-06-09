from pymilvus import MilvusClient
from config import ROOT_PATH

__all__ = [
    'DB_PATH',
    'MILVUS_CLIENT',
    'LAWS_COLLECTION_NAME',
    'BIAODEWU_INFO_COLLECTION_NAME',
    'SENTENCE_EMBEDDING_MODEL_DIM'

]

# Milvus Lite的数据库存放位置(Bid.db)，如果是服务器部署的需要对这块做一些修改
DB_PATH = ROOT_PATH.joinpath('src/vectorstore/db/Bid2.db')

# 对应数据库(Bid.db)的Milvus客户端
MILVUS_CLIENT = MilvusClient(uri=str(DB_PATH))

# 相关法律collection(表)的名称(默认采用'laws_collection')
LAWS_COLLECTION_NAME = 'laws_collection'
# 相关标的物信息collection(表)的名称(默认采用'biaodewu_info_collection')
BIAODEWU_INFO_COLLECTION_NAME = 'biaodewu_info_collection'

# 嵌入模型维度
SENTENCE_EMBEDDING_MODEL_DIM = 1024  # 必须与嵌入模型维度一致！

# collections列表，用于和query cls结果做对应
COLLECTIONS = [LAWS_COLLECTION_NAME, BIAODEWU_INFO_COLLECTION_NAME]
