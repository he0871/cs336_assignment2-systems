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


uv run nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace,autograd-shapes-nvtx--cudabacktrace=all --python-backtrace=cuda --gpu-metrics-devices=0 -- python benchmark.py

Just drop the flag

uv run nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace,autograd-shapes-nvtx--cudabacktrace=all --python-backtrace=cuda --gpu-metrics-devices=0 -- python benchmark.py