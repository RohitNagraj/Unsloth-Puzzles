**Problem:** The problem statement asks toreduce the memory footprint of the last linear layer of a transformer, wherein the output logits matrix shape is `BATCH_SIZE x SEQ_LEN x VOCAB_SIZE`. Given vocab size is over 100k, materializing this matrix can easily take up 4GB at BF16.

**Solution:** Implement gradient checkpointing. Basically, when doing forward pass, use block matrix multiplication and aggregate the loss for each block. During backward pass, recompute the blocks and accumulate the gradients for each block.

The problem asks you to test it with diffent loss functions, thus the `gradient_checkpoint_kld_loss.py` and `gradient_checkpoint_mse_loss.py` implmentations. Output is validated using `torch.allclose`.

## How to run
Just run `python benchmark.py`