from typing import List
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from src.model_backend.base_model import BaseModel


class SentenceModel(BaseModel):
    def __init__(self, model_path, model_name=None) -> None:
        super().__init__(model_path, model_name)

    def _load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_path)

    def _load_model(self):
        return AutoModel.from_pretrained(self.model_path)

    def generate(self, prompt: str, *args, **kwargs):
        pass

    def encode(self, sentences: str | List[str], *args, **kwargs):
        """句子嵌入模型自定义方法，用于返回句子嵌入后的结果"""

        # Tokenize the input texts
        batch_dict = self.tokenizer(sentences, max_length=512, padding=True, truncation=True, return_tensors='pt')

        outputs = self.model(**batch_dict)
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings, embeddings.shape


if __name__ == '__main__':
    # test model
    from utils.common_utils import find_project_root

    project_path = find_project_root()

    # ---- Sentence embedding模型 ----
    sentence_model = SentenceModel(
        project_path.joinpath('models/BAAI/bge-large-zh-v1.5'),
        model_name='bge'
    )

    # emd, shape = sentence_model.sentence_embedding('我们爱祖国！')
    emd, shape = sentence_model.encode(
        ['我们爱祖国！', '祖国爱我们！']
    )
    print(emd.shape)
