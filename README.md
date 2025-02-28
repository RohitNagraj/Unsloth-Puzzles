# UnslothPuzzles

Name: Rohit Nagraj

GPU Used for Testing: NVIDIA T4 (And RTX 3090 for BF16 correctness for challenge A)

## Solutions
### Challenge A
✅ Single Triton Kernel  
✅ Speed >= 1.15  (1.40x speedup achieved)  
✅ Kernel works in torch.compile  
✅ Custom asm works (But I did not need to use it. Of course it could have helped speedup even more).  
✅ Uses cache eviction  
✅ Tested in FP16 and BF16 (BF16 on personal RTX 3090)  

### Challenge E

✅ VRAM 50% reduction (can reduce even more if lower chunk_size is used)  
✅ Show cross-entropy loss works  
✅ Show other loss functions works  
✅ Allow dynamic chunk sizes  
✅ Llama 3.2 1B training loss values match across all steps
❌ Works with GRPO Loss kernel (not tested)

### Challenge B

❌ Unsuccessful Attempt

### Challenge C & D

❌ No Attempt
