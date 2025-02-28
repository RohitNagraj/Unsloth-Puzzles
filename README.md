# UnslothPuzzles

Name: Rohit Nagraj

GPU Used for Testing: NVIDIA T4 (And RTX 3090 for BF16 correctness for challenge A)

## Submission
I have one notebook that has the solutions for challenge A and challenge E.

[▶️ COLAB NOTEBOOK](https://colab.research.google.com/drive/1ZV3Ll4vU92quNQDnwPAnVXX11bGP1FgU?usp=sharing)

[▶️ GITHUB LINK OF THE SAME NOTEBOOK](https://github.com/RohitNagraj/UnslothPuzzles/blob/main/Rohit_Unsloth_Puzzles.ipynb)

## Solutions
### Challenge A
✅ Single Triton Kernel  
✅ Speed >= 1.15  
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

![Llama 3.2 1B Training Loss Curves](resources/q5/llama32_1b_loss_curve.png "Llama 3.2 1B Training Loss Curves")

### Challenge B

❌ Unsuccessful Attempt

### Challenge C & D

❌ No Attempt
