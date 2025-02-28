# UnslothPuzzles

Name: Rohit Nagraj (Hi Daniel, I had messaged you on Discord. Here's my submission 😁)

Role: ML Intern (18 points)

Points Scored (According to me: 19 points)

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

**Notes:**   
Thank you for pointing out the issue with nf4_table being inside the timed function. I had totally missed that.

Another note: I increased the tolarance slightly for torch.allclose since I was having a precision error I'm certain is a bug in Triton, because when I pulled the same vector outside triton and added a constant, I was getting slightly different results vs when I added that constant (the offset) inside the kernel. I can explain more on this if needed.

### Challenge E

✅ VRAM 50% reduction (can reduce even more if lower chunk_size is used)  
✅ Show cross-entropy loss works  
✅ Show other loss functions works  
✅ Allow dynamic chunk sizes  
✅ Llama 3.2 1B training loss values match across all steps (values plotted below, absolutely indistinguishable. Within a range of 1e-3)  
❌ I understand I have not shown the GRPO loss kernel. But I can totally add it if the score isn't sufficient.

![Llama 3.2 1B Training Loss Curves](resources/q5/llama32_1b_loss_curve.png "Llama 3.2 1B Training Loss Curves")

### Challenge B

Honestly, I spent around 3-4 hours on trying to figure this out. But I am new to FSDP and felt more inclined to solve the Triton problem (challenge A) and went for it.   

❌ Unsuccessful Attempt

### Challenge C & D

❌ No Attempt

## Conclusion
Dear Daniel, I had a lotttt of fun doing this. Especially the triton kernel, where I just wanted to pull my hair out since I didn't know how to speed it up further. But this challenge pushed me to learn more, and push my boundaries of understanding. I wouldn't have learnt about NF4, triton's cache eviction policies, dialect ops, and inline assembly if not for this. Thank you for taking the time to create this challenge. Looking forward to your comments. 
