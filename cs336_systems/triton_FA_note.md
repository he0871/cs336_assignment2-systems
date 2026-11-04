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


gcloud compute ssh a2t4 \
    --project=sunny-mark-meal-plan-491103 \
    --zone=us-east1-c

git clone https://github.com/he0871/cs336_assignment2-systems.git

pip install uv

echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

cd cs336_assignment2-systems/

uv venv
uv pip install google-cloud-storage
uv run nsys profile -- python benchmark.py

uv run pytest tests/test_attention.py -k triton -v


# references:
https://github.com/onnxsim/onnxsim/blob/0f76c951ebb16024ee40bbecdd7f8b36500072aa/onnxsim/_triton_w4a16.py#L96