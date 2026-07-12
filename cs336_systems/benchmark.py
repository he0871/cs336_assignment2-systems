import os
import timeit
import yaml
import numpy as np
import torch
import torch.nn.functional as F

import cs336_basics.model as model
import cs336_basics.optimizer as optimizer


def load_config():
    with open("hyconfig.yaml") as f:
        return yaml.safe_load(f)


def resolve_dataset_path(path, cache_dir="/tmp/cs336_data"):
    """Return a local path for `path`, downloading it first if it lives in GCS.

    np.memmap requires a real local file, so a gs:// object is fetched once
    into a local cache and reused on subsequent runs.
    """
    if not path.startswith("gs://"):
        return os.path.expanduser(path)

    from google.cloud import storage

    bucket_name, blob_name = path[len("gs://"):].split("/", 1)
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, os.path.basename(blob_name))

    if not os.path.exists(local_path):
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_name)
        blob.download_to_filename(local_path)

    return local_path


def build_model(config):
    device = torch.device(config["device"])

    net = model.BasicsTransformerLM(
        config["vocab_size"],
        config["context_length"],
        config["d_model"],
        config["num_layers"],
        config["num_heads"],
        config["d_ff"],
        config["rope_theta"],
    ).to(device)

    opt = optimizer.AdamW(
        net.parameters(),
        lr=float(config["learning_rate"]),
    )

    return net, opt, device


def prepare_batch(dataset, config, device):
    context_length = config["context_length"]
    batch_size = config["batch_size"]

    starts = np.random.randint(
        0,
        len(dataset) - context_length - 1,
        size=batch_size,
    )

    x = torch.from_numpy(
        np.stack([dataset[s:s + context_length] for s in starts])
    ).long().to(device)

    y = torch.from_numpy(
        np.stack([dataset[s + 1:s + context_length + 1] for s in starts])
    ).long().to(device)

    return x, y


def train_step(model, optimizer, dataset, config, device):
    x, y = prepare_batch(dataset, config, device)

    optimizer.zero_grad()

    logits = model(x)

    loss = F.cross_entropy(
        logits.view(-1, config["vocab_size"]),
        y.view(-1),
    )

    loss.backward()
    optimizer.step()

    return loss.item()


if __name__ == "__main__":
    config = load_config()

    model, optimizer, device = build_model(config)

    print(f"using device: {device}")

    dataset = np.memmap(
        resolve_dataset_path(config["dataset_path"]),
        dtype=np.uint16,
        mode="r",
    )

    # Warm-up (important for PyTorch/MPS/CUDA)
    for _ in range(config["warmup_steps"]):
        train_step(model, optimizer, dataset, config, device)

    timer = timeit.Timer(
        lambda: train_step(
            model,
            optimizer,
            dataset,
            config,
            device,
        )
    )

    repeat = 10
    times = timer.repeat(repeat=repeat, number=1)

    print(f"Average: {sum(times)/repeat:.6f} sec")
    print(f"Min:     {min(times):.6f} sec")
    print(f"Max:     {max(times):.6f} sec")

    print("\nIndividual runs:")
    for i, t in enumerate(times, 1):
        print(f"{i:2d}: {t:.6f} sec")