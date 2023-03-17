# 这个文件是将 muvtuber_chatbot_api 的框架作用于 t5_demo.py 产生的。
# 可以直接运行：REPL or --muvtuber-grpc-service

import os
import torch
from tokenizer import T5PegasusTokenizer
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
import muvtuber_chatbot_api

_this_dir = os.path.dirname(os.path.realpath(__file__))

pretrained_tokenizer_model_path = os.path.join(
    _this_dir, "model", "pretrained-imxly-t5-pegasus-small")


class T5ChatbotConfig(muvtuber_chatbot_api.ChatbotConfig):
    def model_path(self):
        """self.model="chat" => ./model/chat.pt"""
        return os.path.join(_this_dir, "model", self.model + ".pt")


class T5Chatbot(muvtuber_chatbot_api.Chatbot):
    def __init__(self, config: T5ChatbotConfig) -> None:
        super().__init__()

        self.tokenizer = T5PegasusTokenizer.from_pretrained(
            pretrained_tokenizer_model_path)

        self.model = torch.load(config.model_path())
        self.device = torch.device("cpu")  # 不要用 mps，用 mps 更慢且效果巨差
        self.model.to(self.device)
        self.model.eval()

    def ask(self, session_id, prompt, **kwargs):
        ids = self.tokenizer.encode(
            prompt, return_tensors='pt').to(self.device)
        output = self.model.generate(ids,
                                     decoder_start_token_id=self.tokenizer.cls_token_id,
                                     eos_token_id=self.tokenizer.sep_token_id,
                                     max_length=30).cpu()
        output = output.numpy()[0]

        response_text = ''.join(self.tokenizer.decode(
            output[1:-1])  # [CLS] xxx [SEP]
        ).replace(' ', '')
        return response_text


class T5ChatbotFactory(muvtuber_chatbot_api.ChatbotFactory):
    def create_chatbot(self, config: T5ChatbotConfig):
        return T5Chatbot(config)


if __name__ == '__main__':
    chatbot = T5Chatbot(T5ChatbotConfig(
        model="chat3",  # model/chat.pt
    ))

    while True:
        print(chatbot.ask('', input("> ")))
