import os
import deepspeed
import torch
from llmtuner import run_exp


def main():
#     ############################
#     if 0 == LOCAL_RANK:
#         print("*****************start cp pretrain model*****************************")
#         os.system("chmod +x ./s5cmd")
#         os.system("./s5cmd sync {0} {1}".format(os.environ['MODEL_S3_PATH'], os.environ['MODEL_LOCAL_PATH']))
#         print(f'------rank {LOCAL_RANK}, {WORLD_RANK} finished cp-------')

#     torch.distributed.barrier()
#     ############################
    run_exp()
#     ############################
#     if WORLD_RANK == 0:
#         print("*****************finished training, start cp finetuned model*****************************")
#         os.system("./s5cmd sync {0} {1}".format('/tmp/finetuned_model', os.environ['OUTPUT_MODEL_S3_PATH']))
#         print(f'-----finished cp-------')

#     torch.distributed.barrier()
#     ############################

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    # ############################
    # LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    # WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    # WORLD_RANK = int(os.environ['RANK'])
    # deepspeed.init_distributed(dist_backend='nccl', rank=WORLD_RANK, world_size=WORLD_SIZE)
    # ############################
    main()
