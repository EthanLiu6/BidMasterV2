from abc import ABC, abstractmethod


class BaseModel(ABC):

    def __init__(self, model_path, model_name=None) -> None:
        self.model_path = model_path
        self.model_name = model_name
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

    @abstractmethod
    def _load_tokenizer(self):
        # 分词器加载
        pass

    @abstractmethod
    def _load_model(self):
        # 模型加载
        pass

    @abstractmethod
    def generate(self, prompt: str, *args, **kwargs):
        """输入提示语，输出生成的文本，依据需求增加所需参数"""
        pass
