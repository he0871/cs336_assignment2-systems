# test 1 

## config

```yaml
learning_rate: 1e-3
batch_size: 4
context_length: 256
device: "cpu"
load_size: 1000
vocab_size: 10000 # 10k vocab size for tiny stories, 32k vocab size for openwebtext
d_model: 512
num_layers: 4
num_heads: 16
d_ff: 1344
rope_theta: 10000.0
temperature: 0.8
top_p: 0.9
warmup_steps: 5
chunk_size: 200000
measurement_steps: 10


vocab_path: "gs://cs336-data/TinyStoriesV2-GPT4-train-10k.vocab"
merges_path: "gs://cs336-data/TinyStoriesV2-GPT4-train-10k.merges"
checkpoint: "gs://cs336-data/tinyStories_best.pt"
dataset_path: "gs://cs336-data/tinystories_train_encoded.txt"

max_new_tokens: 100
```
## result 
Collecting data...
Average: 2.405525 sec
Min:     2.342750 sec
Max:     2.487058 sec

Individual runs:
 1: 2.384370 sec
 2: 2.487058 sec
 3: 2.486042 sec
 4: 2.418867 sec
 5: 2.351247 sec
 6: 2.342750 sec
 7: 2.416078 sec
 8: 2.370371 sec
 9: 2.403965 sec
10: 2.394500 sec