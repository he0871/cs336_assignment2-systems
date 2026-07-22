# Matrix multiplication
## Flops
2BHT^2d_h​
2(4)(16)(2562)(32)=268,435,456

## Duration
:aten::mul    6600  152186581  23058.5728787879

# Softmax
## Flops
BHT^2
4(16)(2562)=4,194,304

## Durations
:LogSoftmaxBackward0    15    7499173    499944.866666667
:aten::log_softmax   15    6623274    441551.6        
:aten::_log_softmax  15    6623274    441551.6   

# FLOPs ratio
softmax / mul = 4,194,304 / 268,435,456 = 0.015625

# Duration ratio
softmax / mul = 441551 / 23058 = 19.1
