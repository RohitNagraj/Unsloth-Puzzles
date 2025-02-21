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


if __name__ == "__main__":
    torch.manual_seed(42)
    bsz, qlen, hd, vocab = 4, 4096, 4096, 128 * (2**10)
    # bsz, qlen, hd, vocab = 4, 4096, 4096, 64 * (2**10)
    
    
    X_standard = torch.randn(bsz, qlen, hd, requires_grad=True, device=device)
    X_memory = X_standard.clone().detach().requires_grad_()

    labels = torch.randint(0, vocab, (bsz, qlen), device=device)

    # Two identical linear layers.
    linear_standard = nn.Linear(hd, vocab, device=device)
    linear_memory = nn.Linear(hd, vocab, device=device)
    linear_memory.load_state_dict(linear_standard.state_dict())

    # Standard implementation.
    print("Standard implementation:")
    loss_standard = standard_forward(X_standard, linear_standard, labels)
    loss_standard.backward()
    grad_X_standard = X_standard.grad.clone()

    print("Standard Loss:         ", loss_standard.item())
