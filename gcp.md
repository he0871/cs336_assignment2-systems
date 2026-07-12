brew install --cask google-cloud-sdk

gcloud init

gcloud storage cp ~/code/assignment1-basics/data/tinystories_train_encoded.txt gs://cs336-data/
gcloud storage cp ~/code/assignment1-basics/cs336_basics/tokenizers/TinyStoriesV2-GPT4-train-10k.vocab gs://cs336-data/
gcloud storage cp ~/code/assignment1-basics/cs336_basics/tokenizers/TinyStoriesV2-GPT4-train-10k.merges gs://cs336-data/
gcloud storage cp ~/code/assignment1-basics/cs336_basics/train_lm/tinyStories_best.pt  gs://cs336-data/

gcloud storage ls gs://cs336-data/  


# create GPU VM
## find a zone support 
```bash
 gcloud compute accelerator-types list \
  --project=sunny-mark-meal-plan-491103 \
  --filter="name=nvidia-tesla-t4" \
  --format="value(zone.basename())"
```

How to Fix: Requesting a Quota Increase
You need to request a limit increase for the GPUS_ALL_REGIONS metric.

Go to Quotas: Open the [IAM & Admin > Quotas  page](https://console.cloud.google.com/iam-admin/quotass) in the Cloud Console.
Filter for the Metric: Search for GPUS_ALL_REGIONS.
Select and Edit: Check the box next to the metric (the one with the global dimension) and click Edit Quotas.
Submit Request:
Set the new limit to at least 1.
In the "Reason" field, you can enter something simple like: "Needed for starting a single NVIDIA T4 Spot VM for deep learning development."


## submit request
gcloud compute instances create a2t4 \
    --project=sunny-mark-meal-plan-491103 \
    --zone=us-east1-c  \
    --machine-type=n1-standard-1 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB


## response
NAME  ZONE        MACHINE_TYPE   PREEMPTIBLE  INTERNAL_IP  EXTERNAL_IP  STATUS
a2t4  us-east1-c  n1-standard-1  true         10.142.0.2   34.74.6.206  RUNNING

# ssh to vm
gcloud compute ssh a2t4 \
    --project=sunny-mark-meal-plan-491103 \
    --zone=us-east1-c

# Download code
git clone https://github.com/he0871/cs336_assignment2-systems.git

pip install uv

echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

uv run python
uv pip install google-cloud-storage
uv run nsys profile -- python benchmark.py


uv run nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace,autograd-
shapes-nvtx --cudabacktrace=all --python-backtrace=cuda --gpu-metrics-devices=0 -- python
benchmark.py

gcloud compute  scp a2t4:/home/jingyuanhe/cs336_assignment2-systems/cs336_systems/report2.nsys-rep . --zone=us-east1-c


uv run nsys profile \
    --trace=cuda,cudnn,cublas,osrt,nvtx \
    --pytorch=autograd-shapes-nvtx \
    --cudabacktrace=all \
    --python-backtrace=cuda \
    -- python benchmark.py





# Troubleshooting
nvidia-smi

