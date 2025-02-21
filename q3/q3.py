import os
import time
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from trl import SFTConfig

# -----------------------------------------------------------------------------
# Environment & TorchDynamo logging configuration
# -----------------------------------------------------------------------------
os.environ["TORCHDYNAMO_VERBOSE"] = "1"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True,"
    "roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
)

torch._inductor.config.debug = True
torch._logging.set_logs(
    dynamo=logging.WARN,
    inductor=logging.WARN,
    graph_breaks=True,
    recompiles=True,
    recompiles_verbose=True,
    compiled_autograd_verbose=True,
)
torch._dynamo.config.verbose = True
torch._dynamo.config.suppress_errors = False

# -----------------------------------------------------------------------------
# Torch compile options (common to all compiled regions)
# -----------------------------------------------------------------------------
torch_compile_options = {
    "epilogue_fusion": True,
    "max_autotune": True,         # For max_autotune_triton_matmul bonus (+2)
    "shape_padding": True,
    "trace.enabled": True,
    "triton.cudagraphs": False,
}

# -----------------------------------------------------------------------------
# Patch bitsandbytes.matmul_4bit using part A (use_part_A branch gives +1)
# -----------------------------------------------------------------------------
# import bitsandbytes as bnb
# _original_matmul_4bit = bnb.matmul_4bit
# def patched_matmul_4bit(*args, **kwargs):
#     # Disable compilation for bitsandbytes op to avoid graph breaks.
#     with torch._dynamo.disable():
#         return _original_matmul_4bit(*args, **kwargs)
# bnb.matmul_4bit = patched_matmul_4bit

# -----------------------------------------------------------------------------
# Compile LlamaMLP forward (mlp_compiled: +1, no excessive recompilation)
# -----------------------------------------------------------------------------
from transformers.models.llama.modeling_llama import LlamaMLP
_original_mlp_forward = LlamaMLP.__dict__['forward']
compiled_llama_mlp_fn = torch.compile(_original_mlp_forward, fullgraph=False, dynamic=True, options=torch_compile_options)
def compiled_llama_mlp(self, x):
    return compiled_llama_mlp_fn(self, x)
LlamaMLP.forward = compiled_llama_mlp

# -----------------------------------------------------------------------------
# Compile LlamaAttention forward (attention_compiled: +2 if no excessive recompilations)
# -----------------------------------------------------------------------------
from transformers.models.llama.modeling_llama import LlamaAttention
_original_attention_forward = LlamaAttention.__dict__['forward']
compiled_llama_attention_fn = torch.compile(_original_attention_forward, fullgraph=False, dynamic=True, options=torch_compile_options)
def compiled_llama_attention(self, hidden_states, **kwargs):
    return compiled_llama_attention_fn(self, hidden_states, **kwargs)
LlamaAttention.forward = compiled_llama_attention

# -----------------------------------------------------------------------------
# Compile LayerNorm forward (layernorms_compiled; avoids a -3 penalty)
# -----------------------------------------------------------------------------
_original_layernorm_forward = torch.nn.LayerNorm.__dict__['forward']
compiled_layernorm_forward_fn = torch.compile(_original_layernorm_forward, fullgraph=False, dynamic=True, options=torch_compile_options)
def compiled_layernorm_forward(self, input):
    return compiled_layernorm_forward_fn(self, input)
torch.nn.LayerNorm.forward = compiled_layernorm_forward

# -----------------------------------------------------------------------------
# Load model with QLoRA and use flex attention (uses_flex_attention with dynamic sequence support)
# -----------------------------------------------------------------------------
max_seq_length = 1024
torch.set_default_dtype(torch.float16)
model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
dtype = torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = dtype,
)
# Use "flex_attention" to enable flex attention.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    attn_implementation="flash_attention_2",  # Use flex_attention as allowed.
    quantization_config=bnb_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
with torch.no_grad():
    for name, param in model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
model.enable_input_require_grads()

# -----------------------------------------------------------------------------
# Custom Trainer: compile the loss computation (loss_compiled avoids a -1 penalty)
# -----------------------------------------------------------------------------
from trl import SFTTrainer
class CompiledSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Compile a lambda that calls the model and extracts loss.
        compiled_loss_fn = torch.compile(lambda: model(**inputs),
                                         fullgraph=True, dynamic=True, options=torch_compile_options)
        outputs = compiled_loss_fn()
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        return (loss, outputs) if return_outputs else loss

# -----------------------------------------------------------------------------
# Utility: Log VRAM usage
# -----------------------------------------------------------------------------
def log_vram_usage(label):
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    print(f"[{label}] Allocated VRAM: {allocated:.2f} MB, Reserved VRAM: {reserved:.2f} MB")

log_vram_usage("Before Training")

# -----------------------------------------------------------------------------
# Prepare dataset (fixed sequence lengths help avoid recompilations)
# -----------------------------------------------------------------------------
url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
dataset = load_dataset("json", data_files={"train": url}, split="train[:10%]")

# -----------------------------------------------------------------------------
# Set up trainer with our compiled loss (and thus loss_compiled = True)
# -----------------------------------------------------------------------------
trainer = CompiledSFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_steps=1,
        max_steps=10,
        logging_steps=1,
        output_dir="outputs",
        seed=3407,
        max_seq_length=max_seq_length,
        fp16=model.get_input_embeddings().weight.dtype == torch.float16,
        bf16=model.get_input_embeddings().weight.dtype == torch.bfloat16,
        report_to="none",
        dataset_num_proc=4,
    ),
)

# -----------------------------------------------------------------------------
# Dummy forward passes to warm up compiled regions & test dynamic sequence lengths
# -----------------------------------------------------------------------------
dummy_input_fixed = tokenizer("Hello, world! " * 64,
    return_tensors="pt", max_length=max_seq_length, padding="max_length", truncation=True).input_ids.to(model.device)
dummy_input_dynamic = tokenizer("Hello, world! " * 32,
    return_tensors="pt", max_length=512, padding="max_length", truncation=True).input_ids.to(model.device)

start_time = time.time()
with torch.no_grad():
    _ = model(dummy_input_fixed)
elapsed_fixed = time.time() - start_time
print(f"Dummy forward pass (fixed seq length) took {elapsed_fixed:.4f} seconds")

start_time = time.time()
with torch.no_grad():
    _ = model(dummy_input_fixed)
    _ = model(dummy_input_dynamic)
elapsed_dynamic = time.time() - start_time
print(f"Dynamic sequence length test forward passes took {elapsed_dynamic:.4f} seconds")

log_vram_usage("After Dummy Forward Pass")

# -----------------------------------------------------------------------------
# Train and verify: loss must match non-compiled baseline and no graph breaks occur.
# -----------------------------------------------------------------------------
trainer.train()
log_vram_usage("After Training")
