from abc import ABC


class BasePrompt(ABC):
    def __init__(self, template: str):
        """
        提示的模版，结合build_prompt使用

        use like:
            qa_template = "请根据以下内容回答问题：\n内容：{context}\n问题：{question}\n回答："
            qa_prompt = BasePrompt(qa_template)
            prompt_text = qa_prompt.build_prompt(
                question="中国的首都是哪里？",
                context="中国是一个位于东亚的国家，其首都是北京。"
            )
        """
        self.template = template

    def build_prompt(self, question: str, context: str | list[str]) -> str:
        """用关键字填充模板，生成 prompt 文本"""
        return self.template.format(question=question, context=context)

    def __repr__(self):
        return f"<{self.__class__.__name__} template={self.template}>"
