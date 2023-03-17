# 这个文件是个基础的推理程序。
# 直接源于 https://github.com/renmada/t5-pegasus-pytorch 的例子，
# 没有加上类什么的。
# 可以直接运行：REPL

from tokenizer import T5PegasusTokenizer
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
import torch

model_path = './model/pretrained-imxly-t5-pegasus-small'
tokenizer = T5PegasusTokenizer.from_pretrained(model_path)

use_fine_tuned = True

if use_fine_tuned:
    model = torch.load("./model/chat/model")
    device = torch.device("cpu")  # 不要用 mps，用 mps 更慢且效果巨差
    model.to(device)
    model.eval()
else:  # use pretrained
    model = MT5ForConditionalGeneration.from_pretrained(model_path)


text = '你好'
while text:
    text = input("> ")
    ids = tokenizer.encode(text, return_tensors='pt')
    if use_fine_tuned:
        ids = ids.to(device)
    output = model.generate(ids,
                            decoder_start_token_id=tokenizer.cls_token_id,
                            eos_token_id=tokenizer.sep_token_id,
                            max_length=30)
    if use_fine_tuned:
        output = output.cpu()

    output = output.numpy()[0]
    print(''.join(tokenizer.decode(
        output[1:-1])  # [CLS] xxx [SEP]
    ).replace(' ', ''))
