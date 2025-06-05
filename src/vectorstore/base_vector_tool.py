from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np


class BaseVectorTool(ABC):
    """
    向量存储的抽象基类
    """

    @abstractmethod
    def add(
            self,
            texts: List[str],
            embeddings: List[np.ndarray],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None
    ) -> None:
        """
        插入文本及其对应向量表示。

        Args:
            texts (List[str]): 要插入的原始文本内容。
            embeddings (List[np.ndarray]): 对应的嵌入向量列表。
            metadatas (Optional[List[dict]]): 可选的元数据列表，
                                              主要是与每条文本或向量相关联的附加信息，
                                              不用与向量存储，主要用于检索、过滤、展示等操。
            ids (Optional[List[str]]): 可选的唯一标识符列表。
        """
        pass

    @abstractmethod
    def similarity_search(
            self,
            query_embedding: np.ndarray,
            k: int = 5,
            filter: Optional[dict] = None
    ) -> List[Tuple[str, float, dict]]:
        """
        根据给定向量进行相似性搜索，返回最相似的k个结果。

        Args:
            query_embedding (np.ndarray): 查询向量。
            k (int): 返回的结果数量。
            filter (Optional[dict]): 过滤条件（如按元数据过滤）。

        Return:
            可以实现类似返回：
            List[Tuple[str, float, dict]]: 包含文本、相似度分数和元数据的元组列表。
        """
        pass

    def delete(self, ids: List[str]) -> None:
        """
        删除指定ID的条目。

        参数:
            ids (List[str]): 要删除的条目的ID列表。
        """
        raise NotImplementedError("Delete functional is not implemented.")

    def update(
            self,
            texts: List[str],
            embeddings: List[np.ndarray],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None
    ) -> None:
        """
        更新已有的条目/知识数据，不常用，暂时先列出来CURD。

        Args:
            texts (List[str]): 要更新的文本。
            embeddings (List[np.ndarray]): 新的嵌入向量。
            metadatas (Optional[List[dict]]): 新的元数据。
            ids (Optional[List[str]]): 要更新的条目ID。
        """
        raise NotImplementedError("Update functional is not implemented.")

    def save(self, path: str) -> None:
        """
        将当前向量数据库保存到磁盘。

        Args:
            path (str): 保存路径。
        """
        raise NotImplementedError("Save functional is not implemented.")

    def load(self, path: str) -> None:
        """
        从磁盘加载向量数据库。

        Args:
            path (str): 加载路径。
        """
        raise NotImplementedError("Load functional is not implemented.")


if __name__ == '__main__':
    pass
