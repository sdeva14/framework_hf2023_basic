import torch

import collections
import math

import warnings
from itertools import accumulate
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from torch import Tensor
import torch.nn.functional as F

if TYPE_CHECKING:
    from fairseq.modules.multihead_attention import MultiheadAttention

try:
    from amp_C import multi_tensor_l2norm

    multi_tensor_l2norm_available = True
except ImportError:
    multi_tensor_l2norm_available = False

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None

############################

def relu_squared(x: torch.Tensor):
    return F.relu(x).pow(2)

def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
        0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )

def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)

def get_activation_fn(activation: str) -> Callable:
    """Returns the activation function corresponding to `activation`"""

    if activation == "relu":
        return F.relu
    elif activation == "relu_squared":
        return relu_squared
    elif activation == "gelu":
        return gelu
    # elif activation == "gelu_fast":
    #     deprecation_warning(
    #         "--activation-fn=gelu_fast has been renamed to gelu_accurate"
    #     )
    #     return gelu_accurate
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "swish":
        return torch.nn.SiLU
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, collections.OrderedDict):
            # OrderedDict has attributes that needs to be preserved
            od = collections.OrderedDict(
                (key, _apply(value)) for key, value in x.items()
            )
            od.__dict__ = x.__dict__
            return od
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample, device=None):
    device = device or torch.cuda.current_device()

    def _move_to_cuda(tensor):
        # non_blocking is ignored if tensor is not pinned, so we can always set
        # to True (see github.com/PyTorchLightning/pytorch-lightning/issues/620)
        return tensor.to(device=device, non_blocking=True)

    return apply_to_sample(_move_to_cuda, sample)


def move_to_cpu(sample):
    def _move_to_cpu(tensor):
        # PyTorch has poor support for half tensors (float16) on CPU.
        # Move any such tensors to float32.
        if tensor.dtype in {torch.bfloat16, torch.float16}:
            tensor = tensor.to(dtype=torch.float32)
        return tensor.cpu()

    return apply_to_sample(_move_to_cpu, sample)


@torch.no_grad()
def clip_grad_norm_(params, max_norm, aggregate_norm_fn=None) -> torch.Tensor:
    def grad_exists(p):
        return p is not None and getattr(p, "grad", None) is not None

    if isinstance(params, torch.Tensor):
        params = [params]
    params = list(params)
    grads = [
        p.grad.detach() for p in params if grad_exists(p) and not hasattr(p, "expert")
    ]
    expert_grads = [
        p.grad.detach() for p in params if grad_exists(p) and hasattr(p, "expert")
    ]

    if len(grads) == 0:
        if len(params) > 0:
            return params[0].new_tensor(0.0)
        else:
            return torch.tensor(0.0)

    if len(grads) == 1:
        total_norm = torch.norm(grads[0], p=2, dtype=torch.float32)
    else:
        if multi_tensor_l2norm_available:
            total_norm = multi_tensor_total_norm(grads)
        else:
            if torch.cuda.is_available():
                warnings.warn(
                    "amp_C fused kernels unavailable, disabling multi_tensor_l2norm; "
                    "you may get better performance by installing NVIDIA's apex library"
                )
                device = torch.cuda.current_device()
            elif grads[0].device.type == "xla":
                device = grads[0].device
            else:
                device = torch.device("cpu")
            total_norm = torch.norm(
                torch.stack(
                    [torch.norm(g, p=2, dtype=torch.float32).to(device) for g in grads]
                )
            )

    if aggregate_norm_fn is not None:
        total_norm = aggregate_norm_fn(total_norm)

    if max_norm > 0:
        max_norm = float(max_norm)
        clip_coef = (max_norm / (total_norm + 1e-6)).clamp_(max=1)
        torch._foreach_mul_(grads + expert_grads, clip_coef)

    return total_norm
