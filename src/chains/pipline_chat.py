from config.model_config import *
from config.milvus_db_config import *
from config.data_config import *

from src.prompt import BasePrompt
from src.chains import query_cls_chat, bid_laws_chat, bid_info_chat

# model from config
chat_model = CHAT_MODEL
query_analysis_model = QUERY_ANALYSIS_MODEL
stc_emb_model = SENTENCE_EMBEDDING_MODEL

# miluvs database info from config
milvus_client = MILVUS_CLIENT
laws_collection_name = LAWS_COLLECTION_NAME
biaodewu_info_collection_name = BIAODEWU_INFO_COLLECTION_NAME

# data info from config
laws_dir_path = LAWS_DIR_PATH
law_add_file_path = LAW_ADD_FILE_PATH  # None
biaodewu_dir_path = BIAODEWU_DIR_PATH
biaodewu_add_file_path = BIAODEWU_ADD_FILE_PATH  # None


# TODO: 流式输出（chat_model.py的generate方法修改）
class ChatLine:
    def __init__(self):
        self.history = []
        self.queries = []
        self.base_prompt = BasePrompt(
            template='作为一名招投标智能助手，根据用户的问题做出相关解答：\n用户问题：\n{question}\n现有知识：\n{context}'
        )

    def query_cls_and_select_branch(self, query):

        self.queries.append(query)

        query_type = query_cls_chat(
            llm_model=QUERY_ANALYSIS_MODEL,
            query=query,
            thinking=False
        )

        print("当前query类别列表：", QUERY_CLS)
        print("识别query类别：", query_type)
        if query_type not in QUERY_CLS or query_type == QUERY_CLS[-1]:
            self.base_answer(self.base_prompt.build_prompt(
                question=query, context=''
            ))
        else:
            if query_type == QUERY_CLS[0]:  # 招投标法律法规相关
                return bid_laws_chat(
                    llm_model=chat_model,
                    sentence_model=stc_emb_model,
                    milvus_client=milvus_client,
                    query=query,
                    _collection_name=laws_collection_name,
                    thinking=False
                )

            elif query_type == QUERY_CLS[1]:  # 标的物信息相关
                return bid_info_chat(
                    llm_model=chat_model,
                    sentence_model=stc_emb_model,
                    milvus_client=milvus_client,
                    query=query,
                    _collection_name=biaodewu_info_collection_name,
                    thinking=False
                )

    @staticmethod
    def base_answer(process_query):
        return chat_model.generate_unstream(process_query)


if __name__ == '__main__':
    chat_line = ChatLine()
    while True:
        user_input = input("请输入您的问题（输入q退出）：\n")
        if user_input.lower() == 'q':
            break
        response = chat_line.query_cls_and_select_branch(user_input)
        if response:
            print("*" * 30)
            print("回答：\n", response)
        else:
            print("无法处理该问题，请尝试其他问题。")
