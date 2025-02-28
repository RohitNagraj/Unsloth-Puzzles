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
    loss_fn = nn.KLDivLoss(reduction='batchmean')
    print(f"Size of materialized XW: {get_memory_mb(x)} MB")
    x_view = x.view(-1, x.shape[-1])
    loss = loss_fn(x_view, torch.nn.functional.one_hot(labels.view(-1), num_classes=x_view.shape[-1]).float())
    return loss


class MemoryEfficientLinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, linear, labels, forward_function, chunk_size):
        """
            chunk_size: How many sequences do you want to process at once as a batch. 
        """
        ctx.linear = linear
        ctx.labels = labels
        ctx.forward_function = forward_function
        ctx.save_for_backward(X)
        ctx.chunk_size = chunk_size

        total_tokens = X.shape[0] * X.shape[1]
        total_loss = 0.0

        for i in range(0, X.shape[0], chunk_size):

            with torch.no_grad():
                X_chunk = X[i: i + chunk_size].detach()
                labels_chunk = labels[i: i + chunk_size]

            num_tokens_chunk = X_chunk.shape[0] * X_chunk.shape[1]
            loss_chunk = forward_function(
                X_chunk, linear, labels_chunk).to(device)
            loss_chunk = loss_chunk.double()
            total_loss += loss_chunk * num_tokens_chunk

        final_loss = total_loss / total_tokens
        return final_loss.float()

    @staticmethod
    def backward(ctx, d_loss):
        X, = ctx.saved_tensors
        linear = ctx.linear
        labels = ctx.labels
        forward_function = ctx.forward_function
        chunk_size = ctx.chunk_size

        # We accumulate gradients for X.
        grad_X = torch.zeros_like(X)
        # Use chunking (here chunk size is 1; adjust as needed for memory)
        batch_size = X.shape[0]

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

        return grad_X, None, None, None, None


if __name__ == "__main__":
    torch.manual_seed(42)
    # bsz, qlen, hd, vocab = 4, 4096, 4096, 128 * (2**10)
    bsz, qlen, hd, vocab = 4, 4096, 4096, 64 * (2**10)

    X = torch.randn(bsz, qlen, hd, requires_grad=True, device=device)
    labels = torch.randint(0, vocab, (bsz, qlen), device=device)

    linear = nn.Linear(hd, vocab, device=device)

    loss = MemoryEfficientLinear.apply(
        X, linear, labels, transformation_function, X.shape[0]//4)

    print("Backward pass: ")
    loss.backward()

    print("Memory Efficient Loss: ", loss.item())
