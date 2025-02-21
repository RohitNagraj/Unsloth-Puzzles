from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload, MixedPrecision, BackwardPrefetch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import sys
import torch
import torch.distributed as dist

# --- HELPFUL FUNCTIONS TO UNDO PATCHES ---
def remove_patched_module(package_name):
    modules_to_delete = [name for name in sys.modules if name ==
                         package_name or name.startswith(package_name + ".")]
    for name in modules_to_delete:
        del sys.modules[name]


remove_patched_module("trl")
remove_patched_module("transformers")
remove_patched_module("peft")
remove_patched_module("bitsandbytes")

# --- ENVIRONMENT SETUP ---
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# For Kaggle 2x Tesla T4, we set the visible devices.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,roundup_power2_divisions:[32:256,64:128,256:64,>:32]"

# Distributed settings: if you're launching via torchrun, these will be set automatically.
os.environ["RANK"] = os.environ.get("RANK", "0")
os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "2")
os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
# Make sure LOCAL_RANK is set. torchrun will set it; otherwise, default to 0.
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)


def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        print(f"Distributed process group initialized on device {device}.")


setup_distributed()

# --- IMPORTS ---

# FSDP2 imports (requires PyTorch 2.0+)

# Define auto wrap policy (wrap the LlamaDecoderLayer in a tuple).
def auto_wrap_policy(module, recurse, nonwrapped_numel): return transformer_auto_wrap_policy(
    module, recurse, nonwrapped_numel, transformer_layer_cls=(
        LlamaDecoderLayer,)
)


fsdp_mixed_precision = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)
fsdp_config = {
    "auto_wrap_policy": auto_wrap_policy,
    "mixed_precision": fsdp_mixed_precision,
    "cpu_offload": CPUOffload(offload_params=True),
    "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
}

# --- MODEL AND QUANTIZATION SETUP ---
max_seq_length = 2048
torch.set_default_dtype(torch.float16)
model_name = "unsloth/meta-Llama-3.1-8B-Instruct-bnb-4bit"
dtype = torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=dtype,
)

# Load the pre-quantized model (4-bit via bitsandbytes) on the current device.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": device},
    attn_implementation="sdpa",
    quantization_config=bnb_config,
)

# Prepare the model for k-bit training.
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
# Workaround: unset the quantized flag so adapters can be attached.
model.is_quantized = False

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"

# --- APPLY QLoRA ---
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
# Freeze base parameters except for LoRA parameters.
with torch.no_grad():
    for name, param in model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# --- CONVERT NON-FLOAT PARAMETERS TO BUFFERS ---
def convert_non_float_params_to_buffers(module):
    for name, param in list(module._parameters.items()):
        if param is not None and param.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            module._parameters.pop(name)
            if hasattr(module, name):
                delattr(module, name)
            module.register_buffer(name, param)
    for child in module.children():
        convert_non_float_params_to_buffers(child)


convert_non_float_params_to_buffers(model)

# --- CAST FLOAT32 PARAMETERS TO FLOAT16 ---
def cast_float32_to_float16(module):
    for name, param in module.named_parameters():
        if param is not None and param.dtype == torch.float32:
            param.data = param.data.half()
    for name, buf in module._buffers.items():
        if buf is not None and buf.dtype == torch.float32:
            module._buffers[name] = buf.half()
    for child in module.children():
        cast_float32_to_float16(child)


cast_float32_to_float16(model)

# --- WRAP MODEL WITH FSDP ---
model = FSDP(model, use_orig_params=True, **fsdp_config)

# --- DATASET SETUP ---
url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
dataset = load_dataset("json", data_files={"train": url}, split="train[:10%]")

# Create a simple tokenizer-based collator that doesn't rely on PaddingStrategy
def custom_data_collator(features):
    # Extract the data fields we need
    input_ids = [f["text"] for f in features]
    
    # Tokenize everything with the same settings
    batch = tokenizer(
        input_ids,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    
    # Add labels for LM training
    batch["labels"] = batch["input_ids"].clone()
    
    return batch

# --- SETUP TRAINER ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    data_collator=custom_data_collator,  # Use our custom collator instead
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=1,
        max_steps=10,
        logging_steps=1,
        output_dir="outputs",
        seed=3407,
        max_seq_length=max_seq_length,
        fp16=True,
        bf16=False,
        report_to="none",
        dataset_num_proc=4,
        remove_unused_columns=False,  # make sure all columns pass through
        dataloader_num_workers=0,     # disable multiprocess data loading
    ),
)


def main():
    trainer.train()


if __name__ == "__main__":
    main()