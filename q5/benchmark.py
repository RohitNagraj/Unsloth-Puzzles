import torch
import torch.nn as nn

from q5.gradient_checkpoint import MemoryEfficientLinear, transformation_function
from q5.reference import standard_forward

device = 'cuda'

if __name__ == '__main__':
    torch.manual_seed(42)
    
    bsz, qlen, hd, vocab = 4, 4096, 4096, 64 * (2**10)
    
    X_standard = torch.randn(bsz, qlen, hd, requires_grad=True, device=device)
    X_memory   = X_standard.clone().detach().requires_grad_()
    labels = torch.randint(0, vocab, (bsz, qlen), device=device)

    # Two identical linear layers.
    linear_standard = nn.Linear(hd, vocab, device=device)
    linear_memory   = nn.Linear(hd, vocab, device=device)
    linear_memory.load_state_dict(linear_standard.state_dict())

    # Standard implementation.
    print("Standard implementation: ")
    loss_standard = standard_forward(X_standard, linear_standard, labels)
    loss_standard.backward()
    grad_X_standard = X_standard.grad.clone()

    # Reset gradients before memory–efficient run.
    print("\n\nEfficient implementation: ")
    X_memory.grad = None
    linear_memory.weight.grad = None
    linear_memory.bias.grad = None

    # Memory–efficient implementation.
    loss_memory = MemoryEfficientLinear.apply(X_memory, linear_memory, labels, transformation_function, X_memory.shape[0]//2)
    loss_memory.backward()
    grad_X_memory = X_memory.grad.clone()

    print("\n\nStandard Loss:         ", loss_standard.item())
    print("Memory Efficient Loss: ", loss_memory.item())
    print("Gradients equal:       ", torch.allclose(grad_X_standard, grad_X_memory, atol=1e-4))
    print("Max grad difference:   ", (grad_X_standard - grad_X_memory).abs().max().item())
