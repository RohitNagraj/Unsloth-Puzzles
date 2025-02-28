import torch
import torch.nn as nn
device = 'cuda'


def get_memory_mb(tensor):
    memory_bytes = tensor.numel() * tensor.element_size()
    memory_mb = memory_bytes / (1024 * 1024)
    return memory_mb


def transformation_function(batch, linear, labels):
    # Up–project to large space.
    x = linear(batch).float()
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    print(f"Size of materialized XW: {get_memory_mb(x)} MB")
    loss = loss_fn(x.view(-1, x.shape[-1]), labels.view(-1))
    return loss


def standard_forward(X, linear, labels):
    return transformation_function(X, linear, labels)
