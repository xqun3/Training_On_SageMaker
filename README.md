# Training_On_Sagemaker

本项目主要是用 [LLama-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/3e1bf8325c7ae3ad8e6e3ccc1c644d37030c83ff) 这个框架在 SageMaker 上进行 pretrain/sft/rlhf， 可以方便的进行多机多卡或者单机多卡的训练

```
|-- llama_factory                    # 依赖的LLama Factory 实现代码
|-- multi_nodel_deepspeed_lora.ipynb # 在 SageMaker 提交 Training Job 训练任务的示例代码
`-- README.md
```

# Training Job
如果使用 Training Job，可以参考 [multi_nodel_deepspeed_lora.ipynb](https://github.com/xiaoqunnaws/Training_On_SageMaker/blob/main/multi_nodel_deepspeed_lora.ipynb) 这个 notebook 提交任务训练。跟据需要修改对应的训练机器个数以及类型即可。

```
|-- ac_config.yaml                   # accelerat 配置文件
|-- data                             # 训练数据
|-- entry.py                         # training job 任务入口文件
|-- evaluation                       # LLama-Factory 文件
|-- examples                         # LLama-Factory 文件
|-- pyproject.toml                   # LLama-Factory 文件
|-- requirements.txt                 # 依赖包
|-- s5cmd                            
|-- scripts                          # LLama-Factory 文件
|-- src                              # LLama-Factory 文件
`-- train_script_sagemaker.sh        # 训练启动脚本
```

# Notebook/本地环境
如果使用 Notebook 或者本地机器启动训练任务，直接进入 llama_factory 文件夹，根据需要修改 [train_script_sagemaker.sh](https://github.com/xiaoqunnaws/Training_On_SageMaker/blob/main/llama_factory/train_script_sagemaker.sh) 模型路径即可，其他参数也可根据具体情况进行修改

# 更多参数配置
更多参数配置，可以参考 [LLama-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/3e1bf8325c7ae3ad8e6e3ccc1c644d37030c83ff) 说明进行配置