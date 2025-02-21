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


class MemoryEfficientLinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, linear, labels, forward_function):
        ctx.linear = linear
        ctx.labels = labels
        ctx.forward_function = forward_function
        ctx.save_for_backward(X)
        loss = forward_function(X, linear, labels)
        return loss

    @staticmethod
    def backward(ctx, d_loss):
        X, = ctx.saved_tensors
        linear = ctx.linear
        labels = ctx.labels
        forward_function = ctx.forward_function

        # We accumulate gradients for X.
        grad_X = torch.zeros_like(X)
        # Use chunking (here chunk size is 1; adjust as needed for memory)
        batch_size = X.shape[0]
        chunk_size = 1

        # Also accumulate gradients for linear parameters.
        grad_weight = torch.zeros_like(linear.weight)
        grad_bias = torch.zeros_like(linear.bias)

        # Re-run forward for each chunk under grad mode.
        with torch.enable_grad():
            for i in range(0, batch_size, chunk_size):
                X_chunk = X[i: i + chunk_size].detach().requires_grad_()
                labels_chunk = labels[i: i + chunk_size]
                loss_chunk = forward_function(X_chunk, linear, labels_chunk)
                # Compute gradients w.r.t. X_chunk and linear parameters.
                grads = torch.autograd.grad(
                    loss_chunk, (X_chunk, linear.weight, linear.bias),
                    retain_graph=True, allow_unused=True
                )
                grad_X_chunk, grad_weight_chunk, grad_bias_chunk = grads
                grad_X[i: i + chunk_size] = grad_X_chunk * d_loss
                if grad_weight_chunk is not None:
                    grad_weight += grad_weight_chunk * d_loss
                if grad_bias_chunk is not None:
                    grad_bias += grad_bias_chunk * d_loss

        # Manually accumulate gradients for linear parameters.
        if linear.weight.grad is None:
            linear.weight.grad = grad_weight
        else:
            linear.weight.grad = linear.weight.grad + grad_weight
        if linear.bias.grad is None:
            linear.bias.grad = grad_bias
        else:
            linear.bias.grad = linear.bias.grad + grad_bias

        return grad_X, None, None, None


if __name__ == "__main__":
    torch.manual_seed(42)
    # bsz, qlen, hd, vocab = 4, 4096, 4096, 128 * (2**10)
    bsz, qlen, hd, vocab = 4, 4096, 4096, 64 * (2**10)

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

    # Reset gradients before memory–efficient run.
    X_memory.grad = None
    linear_memory.weight.grad = None
    linear_memory.bias.grad = None

    # Memory–efficient implementation.
    print("Efficient Implementation:")
    loss_memory = MemoryEfficientLinear.apply(
        X_memory, linear_memory, labels, transformation_function)
    print("Backward pass: ")
    loss_memory.backward()
    grad_X_memory = X_memory.grad.clone()

    print("Standard Loss:         ", loss_standard.item())
    print("Memory Efficient Loss: ", loss_memory.item())
    print("Gradients equal:       ", torch.allclose(
        grad_X_standard, grad_X_memory, atol=1e-4))
    print("Max grad difference:   ",
          (grad_X_standard - grad_X_memory).abs().max().item())
