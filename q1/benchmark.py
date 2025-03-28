# The following code is from Unsloth's Puzzles notebook found here: https://colab.research.google.com/drive/1JqKqA1XWeLHvnYAc0wzrR4JBCnq43HyH?usp=sharing#scrollTo=QoE2DGRZG2Ng


from q1.triton_dequant_fused import my_dequantize
from q1.reference_dequant import unsloth_dequantize, peft_dequantize
from transformers.activations import ACT2FN
from bitsandbytes.nn import Linear4bit
from inspect import currentframe as _C, getframeinfo
import torch
import torch.nn as nn
from transformers import set_seed
import time
import inspect
import os

major_version, minor_version = torch.cuda.get_device_capability()
HAS_BFLOAT16 = (major_version >= 8)


def _F(c): return getframeinfo(c).lineno  # Gets line number
def WARN(x): return print(f"\033[31m{x}\033[0m")  # Red colored warnings


# https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string
def NAME(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    names = [var_name for var_name,
             var_val in callers_local_vars if var_val is var]
    return names[0] if len(names) != 0 else ""


def assert_same(x, y, line, dtype):
    assert (x.dtype == dtype)
    try:
        torch.testing.assert_close(x, y, check_stride=True, atol=1e-3, rtol=2)
    except Exception as error:
        raise RuntimeError(
            f"Failed allclose at line [{line}]: {NAME(x)}, {NAME(y)}\n{str(error)}"
        )


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def bnb_Linear4bit(hd, m, dtype=torch.float16):
    return Linear4bit(
        hd, m, bias=None,
        compute_dtype=dtype,
        compress_statistics=True,
        quant_type="nf4",
    )


def assert_correct_bnb(weight, dtype):
    assert (weight.weight.dtype == torch.uint8)
    assert (weight.weight.quant_state.dtype == dtype)
    assert (weight.weight.quant_state.absmax.dtype == torch.uint8)
    assert (weight.weight.quant_state.code.dtype == torch.float32)
    assert (weight.weight.quant_state.offset.dtype == torch.float32)
    assert (weight.weight.quant_state.blocksize == 64)
    assert (weight.weight.quant_state.state2.absmax.dtype == torch.float32)
    assert (weight.weight.quant_state.state2.code.dtype == torch.float32)
    assert (weight.weight.quant_state.state2.blocksize == 256)


class MLP(nn.Module):
    def __init__(self, hd=4096, m=14336, dtype=torch.float16):
        super().__init__()
        self.gate_proj = bnb_Linear4bit(hd, m, dtype=dtype).to("cuda")
        self.up_proj = bnb_Linear4bit(hd, m, dtype=dtype).to("cuda")
        self.down_proj = bnb_Linear4bit(m, hd, dtype=dtype).to("cuda")
        self.gate_proj.weight.quant_state.dtype = dtype
        self.up_proj  .weight.quant_state.dtype = dtype
        self.down_proj.weight.quant_state.dtype = dtype
        self.act_fn = ACT2FN["silu"]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def mlp_forward(X, mlp, fx):
    up = X @ fx(mlp.  up_proj).t()
    gate = X @ fx(mlp.gate_proj).t()
    h = mlp.act_fn(gate) * up
    down = h @ fx(mlp.down_proj).t()
    return down


def mlp_dequantize(X, mlp, fx):
    a = fx(mlp.  up_proj).t()
    torch.cuda.synchronize()
    b = fx(mlp.gate_proj).t()
    torch.cuda.synchronize()
    c = fx(mlp.down_proj).t()
    torch.cuda.synchronize()
    return a, b, c


def test_dequantize(dequantize_fx):
    elapsed = 0
    options = [
        (2, 3333, 2048,  8192, 3407, torch.float16),
        (5,  777, 1024,  4096, 3409, torch.bfloat16),
        (3, 2048, 4096, 14336, 3408, torch.bfloat16),
    ]

    for (bsz, qlen, hd, m, seed, dt) in options:
        set_seed(seed)
        torch.set_default_dtype(torch.float32)
        mlp = MLP(hd=hd, m=m, dtype=dt)
        X = torch.randn((bsz, qlen, hd), device="cuda", dtype=dt)
        torch.cuda.synchronize()

        # Warmup
        for _ in range(2):
            assert_same(mlp_forward(X, mlp, dequantize_fx),
                        mlp(X), _F(_C()), dt)
            assert_correct_bnb(mlp.  up_proj, dt)
            assert_correct_bnb(mlp.gate_proj, dt)
            assert_correct_bnb(mlp.down_proj, dt)
            a, b, c = mlp_dequantize(X, mlp, dequantize_fx)
            A, B, C = mlp_dequantize(X, mlp, unsloth_dequantize)
            assert_same(a, A, _F(_C()), dt)
            assert_same(b, B, _F(_C()), dt)
            assert_same(c, C, _F(_C()), dt)

        # Benchmarking
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(1000):
            mlp_dequantize(X, mlp, dequantize_fx)
        elapsed += time.time() - start
    return elapsed


if __name__ == '__main__':
    unsloth_time = test_dequantize(unsloth_dequantize)
    my_time = test_dequantize(my_dequantize)
    print(f"Unsloth: ", unsloth_time)
    print("My dequantize: ", my_time)
    print(f"Speedup: {100 * (unsloth_time / my_time)}%")
    print("PEFT: ", test_dequantize(peft_dequantize))
