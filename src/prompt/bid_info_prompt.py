from src.prompt.base_prompt import BasePrompt


class BidInfoPrompt(BasePrompt):
    def __init__(self):
        template = '作为一名招投标助手，请根据相关内容回答问题(内容不一定与问题全部相关)：\n内容：\n{context}\n\n问题：\n{question}\n回答并列出具体信息。'
        super().__init__(template=template)


if __name__ == '__main__':
    print(BidInfoPrompt().build_prompt(question='aaa', context='bbb'))
