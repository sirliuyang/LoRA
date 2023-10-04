#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn

from typing import Dict

from .layers import LoRALayer


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    """
    将LoRA模型中的参数设置为可训练或不可训练，可以选择包括偏置（bias）参数。

    参数:
    - model (nn.Module): 要设置参数可训练性的LoRA模型。
    - bias (str, 可选): 决定是否包括偏置参数的选项。
        - 'none': 不包括任何偏置参数。
        - 'all': 包括所有参数，包括偏置参数。
        - 'lora_only': 只包括LoRA相关参数的偏置参数。
    返回:
    - None: 函数没有返回值，而是直接修改了模型的参数的requires_grad属性。
    """

    # 遍历模型的所有命名参数
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            # 如果参数名称中不包含'lora_'，将参数设置为不可训练
            p.requires_grad = False

    # 根据bias参数的不同选项，进一步设置参数的可训练性
    if bias == 'none':
        # 不包括任何偏置参数，直接返回
        return
    elif bias == 'all':
        # 包括所有参数，包括偏置参数，并将它们设置为可训练
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        # 只包括LoRA相关模块的偏置参数，并将它们设置为可训练
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                m.bias.requires_grad = True
    else:
        # 如果bias参数无效，引发NotImplementedError异常
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    """
    获取LoRA模型的状态字典，可以选择包括偏置（bias）参数。

    参数:
    - model (nn.Module): 要获取状态字典的LoRA模型。
    - bias (str, 可选): 决定是否包括偏置参数的选项。
        - 'none': 不包括任何偏置参数。
        - 'all': 包括所有参数，包括偏置参数。
        - 'lora_only': 只包括LoRA相关参数和它们的偏置参数。
    返回:
    - Dict[str, torch.Tensor]: 包含所选参数的状态字典。
    """

    # 获取模型的完整状态字典
    my_state_dict = model.state_dict()

    # 根据bias参数的不同选项进行筛选和返回状态字典
    if bias == 'none':
        # 不包括任何偏置参数
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        # 包括所有参数，包括偏置参数
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        # 只包括LoRA相关参数和它们的偏置参数
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0] + 'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        # 如果bias参数无效，引发NotImplementedError异常
        raise NotImplementedError
