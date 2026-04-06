import torch


def collate_fn(batch):
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    batch_size = len(batch)
    max_tokens = max(item["tokens"].shape[0] for item in batch)
    _, channels, n_mels, chunk_size = batch[0]["tokens"].shape

    padded_tokens = torch.zeros(
        batch_size,
        max_tokens,
        channels,
        n_mels,
        chunk_size,
        dtype=torch.float32,
    )

    token_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool)
    vads = []
    global_features = []

    for i, item in enumerate(batch):
        n_tokens = item["tokens"].shape[0]
        padded_tokens[i, :n_tokens] = item["tokens"]
        token_mask[i, :n_tokens] = True
        vads.append(item["vad"])
        global_features.append(item["global_features"])

    return {
        "tokens": padded_tokens,
        "token_mask": token_mask,
        "vad": torch.stack(vads, dim=0),
        "global_features": torch.stack(global_features, dim=0),
    }