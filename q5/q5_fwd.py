import torch
import torch.nn as nn
device = 'cuda'


class MemoryEfficientLinear(torch.autograd.Function):

    def __init__(self):
        self.bsz, self.qlen, self.hd, self.vocab = 4, 4096, 4096, 128*(2**10)
        # self.bsz, self.qlen, self.hd, self.vocab = 4, 4096, 4096, 4*(2**10)
        self.labels = torch.randint(
            0, self.vocab, (self.bsz, self.qlen)).to(device)
        self.X = torch.randn(self.bsz, self.qlen, self.hd,
                             requires_grad=True).to(device)
        print(f"Size of X: {self.get_memory_mb(self.X)} MB")
        self.linear = nn.Linear(self.hd, self.vocab).to(device)
        print(f"Size of W: {self.get_memory_mb(self.linear.weight)} MB")

    def get_memory_mb(self, tensor):
        memory_bytes = tensor.numel() * tensor.element_size()
        memory_mb = memory_bytes / (1024 * 1024)
        return memory_mb

    def transformation_function(self, batch, linear, labels):
        # Up–project to large space.
        x = linear(batch).float()
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")
        print(f"Size of materialized XW: {self.get_memory_mb(x)} MB")
        loss = self.loss_fn(x.view(-1, x.shape[-1]), labels.view(-1))
        return loss

    def run_standard(self):
        loss = self.transformation_function(
            self.X, self.linear, self.labels).to(device)
        # loss.backward()
        return loss

    def run_optimized(self):

        total_tokens = self.bsz * self.qlen
        total_loss = 0.0

        chunk_size = 2

        for i in range(0, self.X.shape[0], chunk_size):
            X_chunk = self.X[i:i+chunk_size]
            labels_chunk = self.labels[i:i+chunk_size]
            num_tokens_chunk = X_chunk.shape[0] * X_chunk.shape[1]
            loss_chunk = self.transformation_function(
                X_chunk, self.linear, labels_chunk).to(device)
            loss_chunk = loss_chunk.double()
            total_loss += loss_chunk * num_tokens_chunk
        final_loss = total_loss / total_tokens
        return final_loss.float()


if __name__ == "__main__":
    with torch.no_grad():
        forward = Forward()
        loss_standard = forward.run_standard()
        del forward
        torch.cuda.empty_cache()
        forward = Forward()
        loss_memory = forward.run_optimized()
        del forward
        torch.cuda.empty_cache()

        print("Standard Loss:         ", loss_standard.item())
        print("Memory Efficient Loss: ", loss_memory.item())
        print("Gradients equal:       ", torch.allclose(
            loss_standard, loss_memory, atol=1e-4))
        print("Max grad difference:   ",
              (loss_standard - loss_memory).abs().max().item())
