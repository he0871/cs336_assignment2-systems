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


```
E       triton.compiler.errors.CompilationError: at 72:14:
E           o_i = tl.zeros((1, ROW_TILE_SIZE, D), dtype=tl.float32)
E           l_j = tl.zeros((1, ROW_TILE_SIZE, 1), dtype=tl.float32)
E           o_j = tl.zeros((1, ROW_TILE_SIZE, D), dtype=tl.float32)
E           for i in range(tl.cdiv(NUM_KEYS, COLUMN_TILE_SIZE)):
E       
E               k_block = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
E               v_block = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
E               k_transposed = tl.trans(k_block, (0,2, 1))
E               s_j = tl.dot(q_block, k_transposed)
E               s_j = s_j / (D ** 0.5)
E               prev_m = m_j
E               m_j = tl.dot(prev_m, tl.max(s_j, axis=2, keep_dims=True))
E                     ^
E       input and other must have equal reduction dimensions
```
