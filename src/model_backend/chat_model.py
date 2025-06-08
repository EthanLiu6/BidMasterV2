from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model_backend.base_model import BaseModel
import torch

class Qwen3Model(BaseModel):

    def __init__(self, model_path, model_name='Qwen3') -> None:
        super().__init__(model_path, model_name)

    def _load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_path)

    def _load_model(self):
        return AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map=self.device
        )

    def generate(self, prompt: str, enable_thinking=True, *args, **kwargs):

        """
            Args:
                prompt: 输入的内容或者提示内容
                enable_thinking: 是否需要是深度思考模式（推理模式）

            Returns:
                if enable_thinking=True: return 思考内容和生成内容
                else:  只 return 生成内容
        """
        messages = [
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking  # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

                    # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=kwargs.get('max_new_tokens', 32768),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                do_sample=True
            )
        # # conduct text completion
        # generated_ids = self.model.generate(
        #     **model_inputs,
        #     max_new_tokens=32768
        # )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        if enable_thinking:
            # parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

            return thinking_content, content

        else:
            # non-think mode processing
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            return content






if __name__ == '__main__':
    # test model
    from utils.common_utils import find_project_root

    project_path = find_project_root()

    # ---- Chat模型 ----
    qwen3_model = Qwen3Model(
        model_path=project_path.joinpath('models/Qwen/Qwen3-0.6B'),
        model_name='Qwen3'
    )

    # think, connect = qwen3_model.generate('给我讲讲招投标')
    # print(think)
    # print(connect)

    connect = qwen3_model.generate('给我讲讲招投标', enable_thinking=False)
    print(connect)
