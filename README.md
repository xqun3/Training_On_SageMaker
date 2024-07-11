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


# deepspeed zero 多级多卡训练

deepspeed zero 介绍

1. 优化器状态分区（ZeRO stage 1）
2. 梯度分区（ZeRO stage 2）
3. 参数分区（ZeRO stage 3）
4. ZeRO-Offload 到 CPU 和 NVMe

zero1 和 zero2 训练的模型不需要特别处理，zero3 在训练过程中因为会切分参数，所以训练完成后需要合并权重

## 训练直接存储fp16方式

修改 ac_config.yaml 中 deepspeed plugin 参数
```
deepspeed_config:
deepspeed_multinode_launcher: standard
gradient_clipping: 1.0
offload_optimizer_device: none
offload_param_device: none
zero3_init_flag: true
zero_stage: 3
zero3_save_16bit_model: true
```

## 离线合并方式

[官方文档](https://huggingface.co/docs/accelerate/usage_guides/deepspeed#saving-and-loading)：
```
# 进入到需要合并权重的 ckpt 文件夹
cd output_dir/checkpoint-1/

# 有个自动转 fp32 的文件，运行后会生成一个 pytorch_model.bin 的文件
python zero_to_fp32.py . pytorch_model.bin

# 如果需要 shard 成多个模型文件，有两种方式
# 1. 修改 zero_to_fp32.py 文件直接转存
# 2. 使用 transformers load 模型后再使用 save_pretrained 存储

from transformers AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("output_dir/checkpoint-1/", device_map="auto", torch_dtype=torch.float16)
model.save_pretrained("output_dir/checkpoint-1/merged_model/")

# 将原始的 tokenizer 相关文件复制到 merge 后的文件夹下
cp tokenizer.json tokenizer_config.json special_tokens_map.json merged_model/
```

# 更多参数配置
更多参数配置，可以参考 [LLama-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/3e1bf8325c7ae3ad8e6e3ccc1c644d37030c83ff) 说明进行配置