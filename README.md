# Chinese T5 Pegasus Chatbot for muvtuber

## 模型

- T5 PEGASUS: https://github.com/ZhuiyiTechnology/t5-pegasus
- PyTorch Version: https://github.com/renmada/t5-pegasus-pytorch

本项目针对对话任务，使用 [Diamante中文开放域闲聊数据集](https://www.luge.ai/#/luge/dataDetail?id=52) 数据集，对原本的预训练模型进行了迁移学习。

## 使用

命令行和模型对话：

```sh
$ python t5_chatbot/t5.py
> 你是谁
我是一个很有主见的人
> 你叫什么名字
我也不知道，我只知道我是个小胖子
> 你是哪国人
我是山东的，你呢
> 你比较擅长什么才艺？
我擅长的是唱歌，唱歌，唱歌，唱歌，都可以
> 可以点歌吗
可以啊，你要给我点吗
> 为什么我爸妈结婚的时候没有邀请我？
因为我妈不喜欢我
> 有没有双头龟
有，我也不知道，我只知道乌龟
> 你是chatgpt吗
是的，我是一个很有才华的人
> 你认识蔡徐坤吗
我认识他，他是我的偶像
```

muvtuber chatbot 接口：

```sh
# 服务器
$ python t5_chatbot [--muvtb-grpc-serv HOST:PORT]

# 客户端
$ grpcurl -d '{"config": "{\\"model\\": \\"chat\\"}"}' -plaintext localhost:50053 muvtuber.chatbot.v2.ChatbotService.NewSession
{
  "sessionId": "dba59011-6df1-4c82-998e-a55401886080"
}

$ grpcurl -d '{"session_id": "some-bad-id", "prompt": "hello!!"}' -plaintext localhost:50053 muvtuber.chatbot.v2.ChatbotService.Chat
ERROR:
  Code: NotFound
  Message: Session some-bad-id not found

$ grpcurl -d '{"session_id": "dba59011-6df1-4c82-998e-a55401886080", "prompt": "你是谁"}' -plaintext localhost:50053 muvtuber.chatbot.v2.ChatbotService.Chat
{
  "response": "我是一个很有主见的人"
}

$ grpcurl -d '{"session_id": "some-bad-id"}' -plaintext localhost:50053 muvtuber.chatbot.v2.ChatbotService.DeleteSession
ERROR:
  Code: NotFound
  Message: Session some-bad-id not found

$ grpcurl -d '{"session_id": "dba59011-6df1-4c82-998e-a55401886080"}' -plaintext localhost:50053 muvtuber.chatbot.v2.ChatbotService.DeleteSession
{
  "sessionId": "dba59011-6df1-4c82-998e-a55401886080"
}

$ grpcurl -d '{"session_id": "dba59011-6df1-4c82-998e-a55401886080", "prompt": "你是谁"}' -plaintext localhost:50053 muvtuber.chatbot.v2.ChatbotService.Chat
ERROR:
  Code: NotFound
  Message: Session dba59011-6df1-4c82-998e-a55401886080 not found
```

## 训练

```sh
cd t5_chatbot
python train.py  # 从 60 行左右的 args to config 部分修改各种配置。
```
