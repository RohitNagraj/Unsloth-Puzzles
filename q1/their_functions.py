from unsloth.kernels.utils import fast_dequantize

from peft.utils.integrations import dequantize_module_weight as peft_dequantize


def unsloth_dequantize(weight):
    return fast_dequantize(weight.weight, weight.weight.quant_state)
