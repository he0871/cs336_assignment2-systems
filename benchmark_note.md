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
```
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

nsys stats --report nvtx_gpu_proj_trace report3.nsys-rep 

# check forward 
nsys stats --report nvtx_gpu_proj_trace --filter-nvtx=Forward report3.nsys-rep


# runtime comparison 
Compare the runtime of the softmax operation versus the matrix multiplication operations
within the self-attention layer of your model during a forward pass. How does the difference
in runtimes compare to the difference in FLOPs?

## install  sqlite
brew install sqlite

# Download sqlite file
gcloud compute  scp a2t4:/home/jingyuanhe/cs336_assignment2-systems/cs336_systems/report3.sqlite . --zone=us-east1-c
gcloud compute  scp a2t4:/home/jingyuanhe/cs336_assignment2-systems/cs336_systems/proj_nvtx_gpu_proj_trace.csv . --zone=us-east1-c

nsys stats --report nvtx_gpu_proj_trace --filter-nvtx=Forward report3.nsys-rep

## Query desirable columns
```sql
sqlite3 proj.db <<'SQL'
.headers on
.mode column
SELECT "Name",
       CAST("Projected Duration (ns)" AS INTEGER) AS dur_ns,
       "NumGPUOps"
FROM t
WHERE "Name" LIKE '%aten::_log_softmax%'
   OR "Name" LIKE '%aten::mul%'
ORDER BY dur_ns DESC;
SQL
```

result
_log_softmax_backward_data duration:507868 NumGPUOps:1
:aten::mul duration:206430 NumGPUOps:1


## Aggregate by op type (the real breakdown)
```sql
sqlite3 proj.db <<'SQL'
.headers on
.mode column
SELECT
  CASE WHEN instr("Name", ',') > 0
       THEN substr("Name", 1, instr("Name", ',') - 1)
       ELSE "Name" END AS op,
  COUNT(*)                                      AS n,
  SUM(CAST("Projected Duration (ns)" AS INTEGER)) AS total_ns,
  AVG(CAST("Projected Duration (ns)" AS INTEGER)) AS avg_ns
FROM t
GROUP BY op
ORDER BY total_ns DESC;
SQL
```
## result
```
------------------------------------------------------------  ----  ---------  ----------------
:Forward                                                      15    768010231  51200682.0666667

:aten::bmm                                                    1665  568596049  341499.128528529

:Optimizer.step#AdamW.step                                    15    371197135  24746475.6666667

:Optimizer                                                    15    371197135  24746475.6666667

:autograd::engine::evaluate_function: BmmBackward0            555   367367732  661923.841441441

:BmmBackward0                                                 555   367367732  661923.841441441

:aten::einsum                                                 555   205059903  369477.302702703

:autograd::engine::evaluate_function: IndexBackward0          15    204219127  13614608.4666667

:IndexBackward0                                               15    204219127  13614608.4666667

:aten::_index_put_impl_                                       15    158080546  10538703.0666667

:aten::mul                                                    6600  152186581  23058.5728787879

:autograd::engine::evaluate_function: DivBackward0            120   88477151   737309.591666667

:DivBackward0                                                 120   83331242   694427.016666667

:aten::div                                                    1215  74148039   61027.1925925926

:aten::where                                                  240   72940616   303919.233333333

:aten::copy_                                                  1120  49936258   44585.9446428571

:autograd::engine::evaluate_function: MulBackward0            870   48256562   55467.3126436782

:autograd::engine::evaluate_function: MaxBackward0            60    45534197   758903.283333333

:aten::add                                                    2250  44298048   19688.0213333333

:MulBackward0                                                 870   36663314   42141.7402298851

:aten::add_                                                   735   33500122   45578.3972789116

:aten::value_selecting_reduction_backward                     60    32804251   546737.516666667

:MaxBackward0                                                 60    32804251   546737.516666667

:aten::sub_                                                   1170  29244138   24994.9897435897

:aten::cross_entropy_loss                                     15    22472045   1498136.33333333

:torch::autograd::AccumulateGrad                              435   21841983   50211.4551724138

:autograd::engine::evaluate_function: torch::autograd::Accum  435   21841983   50211.4551724138
ulateGrad                                                                                      

:aten::sum                                                    450   20866161   46369.2466666667

:aten::fill_                                                  1395  20025485   14355.1863799283

:aten::zero_                                                  1260  19653617   15598.1087301587

:aten::neg                                                    240   18450701   76877.9208333333

:aten::pow                                                    990   15879731   16040.1323232323

:autograd::engine::evaluate_function: SubBackward0            180   15048743   83604.1277777778

:aten::to                                                     70    15006150   214373.571428571

:aten::_to_copy                                               70    15006150   214373.571428571

:autograd::engine::evaluate_function: ExpBackward0            60    12430208   207170.133333333

:ExpBackward0                                                 60    12430208   207170.133333333

:aten::square                                                 585   11967647   20457.5162393162

:aten::zeros_like                                             1170  11500786   9829.73162393162

:aten::sqrt                                                   585   11226568   19190.7145299145

:aten::reshape                                                480   10716595   22326.2395833333

:aten::clone                                                  480   10716595   22326.2395833333

:SubBackward0                                                 180   9995661    55531.45        

:autograd::engine::evaluate_function: PowBackward0            135   9710934    71932.8444444444

:autograd::engine::evaluate_function: WhereBackward0          60    9425624    157093.733333333

:WhereBackward0                                               60    9425624    157093.733333333

:aten::sub                                                    180   9246008    51366.7111111111

:autograd::engine::evaluate_function: SumBackward1            60    8472510    141208.5        

:aten::exp                                                    60    8452734    140878.9        

:autograd::engine::evaluate_function: SigmoidBackward0        60    8425815    140430.25       

:autograd::engine::evaluate_function: LogSoftmaxBackward0     15    7499173    499944.866666667

:aten::_log_softmax_backward_data                             15    7499173    499944.866666667

:LogSoftmaxBackward0                                          15    7499173    499944.866666667

:aten::max                                                    60    6976009    116266.816666667

:PowBackward0                                                 135   6912777    51205.7555555556

:aten::log_softmax                                            15    6623274    441551.6        

:aten::_log_softmax                                           15    6623274    441551.6        

:aten::cat                                                    240   5971436    24880.9833333333

:autograd::engine::evaluate_function: ViewBackward0           240   5636150    23483.9583333333

:ViewBackward0                                                240   5636150    23483.9583333333

:autograd::engine::evaluate_function: UnsqueezeBackward0      180   4614462    25635.9         

:autograd::engine::evaluate_function: RsqrtBackward0          135   4338012    32133.4222222222

:RsqrtBackward0                                               135   4338012    32133.4222222222

:aten::sigmoid_backward                                       60    4164347    69405.7833333333

:SigmoidBackward0                                             60    4164347    69405.7833333333

:aten::zeros                                                  60    4148988    69149.8         

:aten::concat                                                 120   3369351    28077.925       

:autograd::engine::evaluate_function: NllLossBackward0        15    2934252    195616.8        

:aten::nll_loss_backward                                      15    2934252    195616.8        

:NllLossBackward0                                             15    2934252    195616.8        

:aten::sigmoid                                                60    2725537    45425.6166666667

:autograd::engine::evaluate_function: UnbindBackward0         120   2602085    21684.0416666667

:aten::stack                                                  120   2602085    21684.0416666667

:UnbindBackward0                                              120   2602085    21684.0416666667

:autograd::engine::evaluate_function: ReshapeAliasBackward0   120   2285009    19041.7416666667

:ReshapeAliasBackward0                                        120   2285009    19041.7416666667

:autograd::engine::evaluate_function: MeanBackward1           135   1603604    11878.5481481481

:MeanBackward1                                                135   1603604    11878.5481481481

:aten::mean                                                   135   1441045    10674.4074074074

:aten::new_zeros                                              15    1342645    89509.6666666667

:aten::scatter_                                               60    1297910    21631.8333333333

:aten::nll_loss_nd                                            15    377854     25190.2666666667

:aten::nll_loss_forward                                       15    377854     25190.2666666667

:aten::nll_loss                                               15    377854     25190.2666666667

:aten::arange                                                 150   359294     2395.29333333333

:aten::rsqrt                                                  135   356575     2641.2962962963 

:aten::scalar_tensor                                          120   332381     2769.84166666667

:aten::ge                                                     60    328987     5483.11666666667

CCCL:cub::DeviceRadixSort                                     15    262397     17493.1333333333

:aten::index                                                  15    178591     11906.0666666667

:aten::remainder                                              30    161472     5382.4          

:aten::div_                                                   30    134846     4494.86666666667

:aten::divide_                                                15    67423      4494.86666666667

:aten::ones_like                                              15    39487      2632.46666666667

:Backward                                                     15    39487      2632.46666666667

:aten::item                                                   15    35616      2374.4          

:aten::_local_scalar_dense                                    15    35616      2374.4   
```

```
 Name                                                  Projected Start (ns)  Projected Duration (ns)  Orig Start (ns)  Orig Duration (ns)   Style   PID   TID   NumGPUOps  Lvl  NumChild  RangeId  ParentId         RangeStack   
```

## softmax
```
:aten::log_softmax, seq = 697, op_id = 2229, sizes = [[1024, 10000], [], []], input_op_ids = [(2226…           11439684713                   472124      11407507150            32271620  PushPop  1495  1495          1    2         1     2232      2231  :860:2231:2232          
 :aten::_log_softmax, seq = 697, op_id = 2231, sizes = [[1024, 10000], [], []], input_op_ids = [(223…           11439684713                   472124      11407529564            32200301  PushPop  1495  1495          1    3         0     2234      2232  :860:2231:2232:2234    
```

## mul
```
nsys stats --report nvtx_gpu_proj_trace --filter-nvtx=Forward report3.nsys-rep | grep mul
 :aten::mul, seq = 5, op_id = 871, sizes = [[4, 256, 512], [4, 256, 1]], input_op_ids = [(867,0), (8…           11193827546                    21248      11174067490            19789343  PushPop  1495  1495          1    1         0      872       860  :860:872                
 :aten::mul, seq = 6, op_id = 872, sizes = [[512], [4, 256, 512]], input_op_ids = [(183,0), (871,0)]            11193975289                    18144      11193918297               59600  PushPop  1495  1495          1    1         0      873       860  :860:873                
 :aten::mul, seq = 55, op_id = 973, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(969,0), …           11211250321                    18207      11211195262               80108  PushPop  1495  1495          1    1         0      976       860  :860:976                
 :aten::mul, seq = 56, op_id = 974, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(971,0), …           11211483663                    14688      11211284652              205661  PushPop  1495  1495          1    1         0      977       860  :860:977                
 :aten::mul, seq = 58, op_id = 976, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(971,0), …           11211671661                    17952      11211642407               31362  PushPop  1495  1495          1    1         0      979       860  :860:979                
 :aten::mul, seq = 59, op_id = 977, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(969,0), …           11211704013                    14624      11211679992               25514  PushPop  1495  1495          1    1         0      980       860  :860:980                
 :aten::mul, seq = 65, op_id = 996, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(992,0), …           11223159155                    17696      11223112963               50687  PushPop  1495  1495          1    1         0      999       860  :860:999                
 :aten::mul, seq = 66, op_id = 997, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(994,0), …           11223201458                    13920      11223171028               32112  PushPop  1495  1495          1    1         0     1000       860  :860:1000               
 :aten::mul, seq = 68, op_id = 999, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(994,0), …           11223290226                    17184      11223267417               24398  PushPop  1495  1495          1    1         0     1002       860  :860:1002               
 :aten::mul, seq = 69, op_id = 1000, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(992,0),…           11223315570                    14399      11223297686               18940  PushPop  1495  1495          1    1         0     1003       860  :860:1003               
 :aten::mul, seq = 125, op_id = 1117, sizes = [[4, 256, 512], [4, 256, 1]], input_op_ids = [(1113,0)…           11373008854                    20576      11372979204               32285  PushPop  1495  1495          1    1         0     1120       860  :860:1120               
 :aten::mul, seq = 126, op_id = 1118, sizes = [[512], [4, 256, 512]], input_op_ids = [(188,0), (1117…           11373085269                    19264      11373025585               70110  PushPop  1495  1495          1    1         0     1121       860  :860:1121               
 :aten::mul, seq = 141, op_id = 1145, sizes = [[4, 256, 1344], [4, 256, 1344]], input_op_ids = [(112…           11385769969                    70880      11385695852               78489  PushPop  1495  1495          1    1         0     1148       860  :860:1148               
 :aten::mul, seq = 155, op_id = 1170, sizes = [[4, 256, 1344], [4, 256, 1344]], input_op_ids = [(114…           11386698922                    67936      11386145638              358380  PushPop  1495  1495          1    1         0     1173       860  :860:1173               
 :aten::mul, seq = 174, op_id = 1203, sizes = [[4, 256, 512], [4, 256, 1]], input_op_ids = [(1199,0)…           11387450340                    19520      11387254245               60346  PushPop  1495  1495          1    1         0     1206       860  :860:1206               
 :aten::mul, seq = 175, op_id = 1204, sizes = [[512], [4, 256, 512]], input_op_ids = [(312,0), (1203…           11387471140                    19264      11387329550               39591  PushPop  1495  1495          1    1         0     1207       860  :860:1207               
 :aten::mul, seq = 224, op_id = 1305, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(1301,0…           11389087031                    18400      11389041147               59395  PushPop  1495  1495          1    1         0     1308       860  :860:1308               
 :aten::mul, seq = 225, op_id = 1306, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(1303,0…           11389132887                    14240      11389107887               26641  PushPop  1495  1495          1    1         0     1309       860  :860:1309               
 :aten::mul, seq = 227, op_id = 1308, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(1303,0…           11389201046                    17248      11389181501               20888  PushPop  1495  1495          1    1         0     1311       860  :860:1311               
 :aten::mul, seq = 228, op_id = 1309, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(1301,0…           11389673363                    14752      11389628616               49252  PushPop  1495  1495          1    1         0     1312       860  :860:1312               
 :aten::mul, seq = 234, op_id = 1328, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(1324,0…           11390018768                    17248      11389984486               37370  PushPop  1495  1495          1    1         0     1331       860  :860:1331               
 :aten::mul, seq = 235, op_id = 1329, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(1326,0…           11390053200                    14688      11390027901               26604  PushPop  1495  1495          1    1         0     1332       860  :860:1332               
 :aten::mul, seq = 237, op_id = 1331, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(1326,0…           11390117903                    17088      11390091948               26748  PushPop  1495  1495          1    1         0     1334       860  :860:1334               
 :aten::mul, seq = 238, op_id = 1332, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(1324,0…           11390140623                    14144      11390123802               17638  PushPop  1495  1495          1    1         0     1335       860  :860:1335               
 :aten::mul, seq = 294, op_id = 1449, sizes = [[4, 256, 512], [4, 256, 1]], input_op_ids = [(1445,0)…           11392977785                    19776      11392942604               21820  PushPop  1495  1495          1    1         0     1452       860  :860:1452               
 :aten::mul, seq = 295, op_id = 1450, sizes = [[512], [4, 256, 512]], input_op_ids = [(317,0), (1449…           11392998969                    19423      11392973127               29932  PushPop  1495  1495          1    1         0     1453       860  :860:1453               
 :aten::mul, seq = 310, op_id = 1477, sizes = [[4, 256, 1344], [4, 256, 1344]], input_op_ids = [(145…           11393913841                    70400      11393392652               26173  PushPop  1495  1495          1    1         0     1480       860  :860:1480               
 :aten::mul, seq = 324, op_id = 1502, sizes = [[4, 256, 1344], [4, 256, 1344]], input_op_ids = [(147…           11394597356                    69119      11393684250               30328  PushPop  1495  1495          1    1         0     1505       860  :860:1505               
 :aten::mul, seq = 343, op_id = 1535, sizes = [[4, 256, 512], [4, 256, 1]], input_op_ids = [(1531,0)…           11395317286                    18912      11394523126               22391  PushPop  1495  1495          1    1         0     1538       860  :860:1538               
 :aten::mul, seq = 344, op_id = 1536, sizes = [[512], [4, 256, 512]], input_op_ids = [(441,0), (1535…           11395337510                    19424      11394561065               25050  PushPop  1495  1495          1    1         0     1539       860  :860:1539               
 :aten::mul, seq = 393, op_id = 1637, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(1633,0…           11396165408                    18879      11395639641               42988  PushPop  1495  1495          1    1         0     1640       860  :860:1640               
 :aten::mul, seq = 394, op_id = 1638, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(1635,0…           11396185695                    14368      11395689510               29272  PushPop  1495  1495          1    1         0     1641       860  :860:1641               
 :aten::mul, seq = 396, op_id = 1640, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(1635,0…           11396210143                    16800      11395759575               19891  PushPop  1495  1495          1    1         0     1643       860  :860:1643               
 :aten::mul, seq = 397, op_id = 1641, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(1633,0…           11396228191                    14496      11395784646               17474  PushPop  1495  1495          1    1         0     1644       860  :860:1644               
 :aten::mul, seq = 403, op_id = 1660, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(1656,0…           11396286175                    18239      11396062568               35054  PushPop  1495  1495          1    1         0     1663       860  :860:1663               
 :aten::mul, seq = 404, op_id = 1661, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(1658,0…           11396305630                    14560      11396103470               20666  PushPop  1495  1495          1    1         0     1664       860  :860:1664               
 :aten::mul, seq = 406, op_id = 1663, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(1658,0…           11396329694                    16832      11396178314               20094  PushPop  1495  1495          1    1         0     1666       860  :860:1666               
 :aten::mul, seq = 407, op_id = 1664, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(1656,0…           11396347678                    14528      11396203487               24709  PushPop  1495  1495          1    1         0     1667       860  :860:1667               
 :aten::mul, seq = 463, op_id = 1781, sizes = [[4, 256, 512], [4, 256, 1]], input_op_ids = [(1777,0)…           11399102792                    19712      11399079832               25030  PushPop  1495  1495          1    1         0     1784       860  :860:1784               
 :aten::mul, seq = 464, op_id = 1782, sizes = [[512], [4, 256, 512]], input_op_ids = [(446,0), (1781…           11399137608                    19040      11399114324               24879  PushPop  1495  1495          1    1         0     1785       860  :860:1785               
 :aten::mul, seq = 479, op_id = 1809, sizes = [[4, 256, 1344], [4, 256, 1344]], input_op_ids = [(178…           11400100801                    70271      11399552287               26556  PushPop  1495  1495          1    1         0     1812       860  :860:1812               
 :aten::mul, seq = 493, op_id = 1834, sizes = [[4, 256, 1344], [4, 256, 1344]], input_op_ids = [(180…           11400784123                    68192      11399912606               49599  PushPop  1495  1495          1    1         0     1837       860  :860:1837               
 :aten::mul, seq = 512, op_id = 1867, sizes = [[4, 256, 512], [4, 256, 1]], input_op_ids = [(1863,0)…           11401497846                    19359      11400469085               21121  PushPop  1495  1495          1    1         0     1870       860  :860:1870               
 :aten::mul, seq = 513, op_id = 1868, sizes = [[512], [4, 256, 512]], input_op_ids = [(570,0), (1867…           11401518389                    19392      11400498309               27776  PushPop  1495  1495          1    1         0     1871       860  :860:1871               
 :aten::mul, seq = 562, op_id = 1969, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(1965,0…           11402353103                    19168      11401791607               56330  PushPop  1495  1495          1    1         0     1972       860  :860:1972               
 :aten::mul, seq = 563, op_id = 1970, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(1967,0…           11402373487                    15072      11401854646               39377  PushPop  1495  1495          1    1         0     1973       860  :860:1973               
 :aten::mul, seq = 565, op_id = 1972, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(1967,0…           11402398446                    16928      11401934676               19850  PushPop  1495  1495          1    1         0     1975       860  :860:1975               
 :aten::mul, seq = 566, op_id = 1973, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(1965,0…           11402416590                    14336      11401959244               17733  PushPop  1495  1495          1    1         0     1976       860  :860:1976               
 :aten::mul, seq = 572, op_id = 1992, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(1988,0…           11402711116                    17280      11402675861               38436  PushPop  1495  1495          1    1         0     1995       860  :860:1995               
 :aten::mul, seq = 573, op_id = 1993, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(1990,0…           11402742508                    14368      11402720777               22892  PushPop  1495  1495          1    1         0     1996       860  :860:1996               
 :aten::mul, seq = 575, op_id = 1995, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(1990,0…           11402801323                    17184      11402782224               19786  PushPop  1495  1495          1    1         0     1998       860  :860:1998               
 :aten::mul, seq = 576, op_id = 1996, sizes = [[256, 16], [4, 16, 256, 16]], input_op_ids = [(1988,0…           11402830059                    14688      11402807104               23827  PushPop  1495  1495          1    1         0     1999       860  :860:1999               
 :aten::mul, seq = 632, op_id = 2113, sizes = [[4, 256, 512], [4, 256, 1]], input_op_ids = [(2109,0)…           11405487158                    18944      11405466234               22415  PushPop  1495  1495          1    1         0     2116       860  :860:2116               
 :aten::mul, seq = 633, op_id = 2114, sizes = [[512], [4, 256, 512]], input_op_ids = [(575,0), (2113…           11405525462                    19200      11405497809               29066  PushPop  1495  1495          1    1         0     2117       860  :860:2117               
 :aten::mul, seq = 648, op_id = 2141, sizes = [[4, 256, 1344], [4, 256, 1344]], input_op_ids = [(211…           11406421199                    72479      11405883014               25937  PushPop  1495  1495          1    1         0     2144       860  :860:2144               
 :aten::mul, seq = 662, op_id = 2166, sizes = [[4, 256, 1344], [4, 256, 1344]], input_op_ids = [(214…           11407105801                    68160      11406194192               30773  PushPop  1495  1495          1    1         0     2169       860  :860:2169               
 :aten::mul, seq = 681, op_id = 2199, sizes = [[4, 256, 512], [4, 256, 1]], input_op_ids = [(2195,0)…           11407819972                    19360      11406754239               21964  PushPop  1495  1495          1    1         0     2202       860  :860:2202               
 :aten::mul, seq = 682, op_id = 2200, sizes = [[512], [4, 256, 512]], input_op_ids = [(580,0), (2199…           11407840644                    19711      11406784868               27591  PushPop  1495  1495          1    1         0     2203       860  :860:2203               
jingyuanhe@a2t4:~/cs336_assignment2-systems/cs336_systems$ 
```

## workaround
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
  | sudo tee /etc/modprobe.d/nvidia-profiling.conf

sudo update-initramfs -u
sudo reboot


# Good CUDA kernel wiki 
https://developer.nvidia.com/blog/even-easier-introduction-cuda/


```bash
nsys stats --report nvtx_gpu_proj_trace report.nsys-rep
# or
nsys stats --report nvtx_sum report.nsys-rep
```