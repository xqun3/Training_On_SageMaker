 # Training On SageMaker

本项目主要是用 [LLama-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/3e1bf8325c7ae3ad8e6e3ccc1c644d37030c83ff) 这个框架在 SageMaker 上进行 pretrain/sft/rlhf， 可以方便的进行多机多卡或者单机多卡的训练

```
|-- llama_factory                    # 依赖的LLama Factory 实现代码
|   |-- ac_config.yaml                   # accelerat 配置文件
|   |-- data                             # 训练数据
|   |-- entry.py                         # training job 任务入口文件
|   |-- evaluation                       # LLama-Factory 文件
|   |-- examples                         # LLama-Factory 文件
|   |-- pyproject.toml                   # LLama-Factory 文件
|   |-- requirements.txt                 # 依赖包
|   |-- s5cmd                            
|   |-- scripts                          # LLama-Factory 文件
|   |-- src                              # LLama-Factory 文件
|   `-- train_script_sagemaker.sh        # 训练启动脚本
|-- multi_nodel_deepspeed_lora.ipynb # 在 SageMaker 提交 Training Job 训练任务的示例代码
`-- README.md
```

# Training Job
如果使用 Training Job，可以参考 [multi_nodel_deepspeed_lora.ipynb](https://github.com/xiaoqunnaws/Training_On_SageMaker/blob/main/multi_nodel_deepspeed_lora.ipynb) 这个 notebook 提交任务训练。跟据需要修改对应的训练机器个数以及类型即可。


# Notebook/本地环境
如果使用 Notebook 或者本地机器启动训练任务，直接进入 llama_factory 文件夹，根据需要修改 [train_script_sagemaker.sh](https://github.com/xiaoqunnaws/Training_On_SageMaker/blob/main/llama_factory/train_script_sagemaker.sh) 模型路径即可，其他参数也可根据具体情况进行修改

# 数据准备
数据的格式准备可以参考 LLama-Factory 的数据格式进行准备，这里以准备的 alpaca 数据格式为例，假设我有这样的一个 jsonl 文件：

```
# temp_data.jsonl
{"question": "为芬兰拉普兰撰写一份旅游指南。","answer": "拉普兰是一个广袤而令人叹为观止的地区，位于芬兰北部的北极圈内。它从挪威一直延伸到东边，向西延伸至瑞典。拉普兰充满了各种雪中活动，如滑雪、雪鞋行走、雪地摩托、雪橇和冰雕。还有更多的户外娱乐活动，包括驯鹿冒险、雪鞋行走、哈士奇冒险和钓鱼。如果您选择在拉普兰过夜，可以体验传统的拉普兰住宿，如遮蔽冰屋和荒野中的乡村小屋。在这里，极光也是必看的壮观景象。 拉普兰是一个冬季仙境，充满了乐趣和冒险。无论您是寻求肾上腺素充沛的周末还是难忘的假期，都值得一游。"},
{"question": "重写方程 $y = rac{2x-1}{3}$，以便解出 $x$。","answer": "可以通过将方程两边乘以 3 并加 1 来重写方程，得到 $3y+1 = 2x$。接着可以变形为 $2x = 3y+1$，然后通过将两边除以 2 来解出 $x$，得到 $x = rac{3y+1}{2}$。"}

```

则可以将 temp_data.json 放到 llama_factory/data 目录下，并修改 llama_factory/data/dataset_info.json 文件如下所示

```
# 修改 llama_factory/data/dataset_info.json 文件，添加以下内容，
  "temp_data": {
    "file_name": "temp_data.jsonl",
    "columns": {
      "prompt": "query",
      "response": "answer"
    }
  },
```

然后就可以将 llama_factory/train_script_sagemaker_sft_torchrun.sh 里的 datasetas 指定为 temp_data 用于模型的训练

# 更多参数配置
更多参数配置，可以参考 [LLama-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/3e1bf8325c7ae3ad8e6e3ccc1c644d37030c83ff) 说明进行配置