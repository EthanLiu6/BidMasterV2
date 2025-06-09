from src.prompt.base_prompt import BasePrompt


class QueryAnalysisPrompt(BasePrompt):
    def __init__(self):
        template = '作为一名招投标助手，请将用户的问题先进行相关意图理解，然后根据理解分析出问题所属范围是哪一类：\n用户问题：\n{question}\n问题类别：\n{context}\n直接输出问题类别即可。'
        super().__init__(template=template)


if __name__ == '__main__':
    print(QueryAnalysisPrompt().build_prompt(
        question='招标文件要求中标人提交履约保证金的最高限额是多少？',
        context=['招投标法律法规相关', '招投标信息相关', '与招投标无关问题'])
    )
