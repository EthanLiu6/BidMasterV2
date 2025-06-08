from src.prompt.base_prompt import BasePrompt


class LawPrompt(BasePrompt):
    def __init__(self):
        template = '作为一名招投标法律顾问，请根据以下内容回答问题：\n内容：\n{context}\n\n问题：\n{question}\n'
        super().__init__(template=template)


if __name__ == '__main__':
    print(LawPrompt().build_prompt(question='aaa', context='bbb'))
