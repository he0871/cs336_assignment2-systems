# uv run nsys profile -- python benchmark.py 

```
## result 
### CPU profile
Average: 2.431501 sec
Min:     2.303507 sec
Max:     2.562581 sec

### GPU profile
Average: 0.167349 sec
Min:     0.140146 sec
Max:     0.206749 sec

nsys stats /home/jingyuanhe/cs336_assignment2-systems/cs336_systems/report2.nsys-rep



uv run nsys profile \
    --trace=cuda,cudnn,cublas,osrt,nvtx \
    --pytorch=autograd-shapes-nvtx \
    --cudabacktrace=all \
    --python-backtrace=cuda \
    --gpu-metrics-devices=0 \
    -- python benchmark.py

gcloud compute  scp a2t4:/home/jingyuanhe/cs336_assignment2-systems/cs336_systems/report5.nsys-rep . --zone=us-east1-c

nsys stats --report cuda_gpu_kern_sum /home/jingyuanhe/cs336_assignment2-systems/cs336_systems/report5.nsys-rep


## workaround
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
  | sudo tee /etc/modprobe.d/nvidia-profiling.conf

sudo update-initramfs -u
sudo reboot


# Good CUDA kernel wiki 
https://developer.nvidia.com/blog/even-easier-introduction-cuda/