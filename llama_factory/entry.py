import os
import json
import socket
import yaml

if __name__ == "__main__":
   
    hosts = json.loads(os.environ['SM_HOSTS'])
    current_host = os.environ['SM_CURRENT_HOST']
    host_rank = int(hosts.index(current_host))
    
    #Parse the IP address of the master node in the multiple nodes cluster of SageMaker training.
    master = json.loads(os.environ['SM_TRAINING_ENV'])['master_hostname']
    master_addr = socket.gethostbyname(master)
    
    os.environ['DS_BUILD_FUSED_ADAM'] = '1'
    os.environ['NODE_INDEX'] = str(host_rank)
    os.environ['SM_MASTER'] = str(master)
    os.environ['SM_MASTER_ADDR'] = str(master_addr)
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    
    # backend env config
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['FI_PROVIDER'] = 'efa'
    os.environ['NCCL_PROTO'] = 'simple'
   # os.environ['FI_EFA_USE_DEVICE_RDMA'] = '1'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['HCCL_OVER_OFI'] = '1'
    
    num_machines = int(os.environ["NODE_NUMBER"])
    num_processes = int(os.environ["SM_NUM_GPUS"]) * num_machines
    
    # os.system("wandb disabled")
    
    file_name = './ac_config.yaml'
    with open(file_name) as f:
        doc = yaml.safe_load(f)
    doc['machine_rank'] = host_rank
    doc['main_process_ip'] = str(master_addr)
    doc['num_machines'] = num_machines  # how many intances in this training job
    doc['num_processes'] = num_processes  # how many GPU cards in total
    with open('./ac_config.yaml', 'w') as f:
        yaml.safe_dump(doc, f)
    #invoke the torch launcher shell script.
    #Note: we will use the s5cmd to speed up the uploading model assets to S3.
    os.system("chmod +x ./train_script_sagemaker_sft.sh")
    # os.system("chmod +x ./train_script_sagemaker.sh")
    os.system("chmod +x ./s5cmd")

    print("*****************start cp pretrain model*****************************")
    # os.system("./s5cmd sync {0} {1}".format(os.environ['MODEL_S3_PATH'], os.environ["MODEL_LOCAL_PATH"]))
    os.system("aws s3 cp {0} {1} --recursive".format(os.environ['MODEL_S3_PATH'], os.environ["MODEL_LOCAL_PATH"]))
    print(f'-----finished cp-------')


    os.system("/bin/bash -c ./train_script_sagemaker_sft.sh")

    print("*****************finished training, start cp finetuned model*****************************")
    os.system("./s5cmd sync {0} {1}".format("/tmp/finetuned_model", os.environ['OUTPUT_MODEL_S3_PATH']))
    print(f'-----finished cp-------')