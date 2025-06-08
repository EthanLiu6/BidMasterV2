from src.prompt.base_prompt import BasePrompt


class LawPrompt(BasePrompt):
    def __init__(self):
        template = '作为一名招投标法律顾问，请根据相关法律条文回答问题：\n内容：\n{context}\n\n问题：\n{question}\n简洁明了回答并说明条文内容。'
        super().__init__(template=template)


if __name__ == '__main__':
    print(LawPrompt().build_prompt(question='aaa', context='bbb'))
