from src.model_backend.sentence_model import SentenceModel


def test01():
    stc_model = SentenceModel('../models/thenlper/gte-large-zh')
    emb, _ = stc_model.encode('测试文本，用于测试emb转list。')
    print(emb.tolist())


if __name__ == '__main__':
    test01()
