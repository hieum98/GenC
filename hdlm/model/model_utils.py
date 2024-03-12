import logging
import os
from typing import List, Optional, Tuple, Union
import torch
from transformers import PreTrainedModel


def prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True, 
    layer_norm_names=["layer_norm"], 
    output_embedding_layer_name="lm_head"
):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

        if loaded_in_kbit:
            # cast layer norm in fp32 for stability for 8bit models
            if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
                param.data = param.data.to(torch.float32)

    if loaded_in_kbit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    if hasattr(model, output_embedding_layer_name):
        output_embedding_layer = getattr(model, output_embedding_layer_name)
        input_dtype = output_embedding_layer.weight.dtype

        class CastOutputToFloat(torch.nn.Sequential):
            r"""
            Manually cast to the expected dtype of the lm_head as sometimes there is a final layer norm that is casted
            in fp32

            """

            def forward(self, x):
                return super().forward(x.to(input_dtype)).to(torch.float32)

        setattr(model, output_embedding_layer_name, CastOutputToFloat(output_embedding_layer))

    return model


def find_all_linear_names(model, quantization: Optional[int] = None):
    if quantization is None:
        cls = torch.nn.Linear
    elif quantization == 4:
        from bitsandbytes.nn import Linear4bit

        cls = Linear4bit
    elif quantization == 8:
        from bitsandbytes.nn import Linear8bitLt

        cls = Linear8bitLt
    else:
        raise ValueError(f"Unknown quantization type: {quantization}")

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def get_trainable_parameters(model: PreTrainedModel) -> Tuple[int, int, float]:
    """
    Prints the number of trainable parameters in the model.

    Args:
        model (`PreTrainedModel`):
            The model to print the number of trainable parameters for.

    Returns:
        `Tuple[int, int, float]`:
            The number of trainable parameters, the total number of parameters and the
            percentage of trainable parameters.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param, 100 * trainable_params / all_param


def get_device_map(force_auto_device_map: bool=False, max_memory_MB: int = None) -> Tuple[str, Union[int, List[int]]]:
    if force_auto_device_map:
        if os.environ.get("LOCAL_RANK") is not None:
            # raise ValueError(
            #    "Found DDP environment and force_auto_device_map is set to True, this configuration "
            #    "is not supported. If you want to use DPP, set force_auto_device_map to False, so "
            #    "a copy of the model is loaded in each GPU. If you want the split the model across "
            #    "GPUs (force_auto_device_map=True), do not use DDP (launch your script with "
            #    "pyton -m src/run.py config.json). If you are not in a DDP environment but you see "
            #    "this error, you might have manually set the environment variable 'LOCAL_WORLD_SIZE' to a "
            #    "number different than 1, please, remove this environment variable or set it to 1"
            # )
            if torch.cuda.is_available():
                n_gpus = torch.cuda.device_count()
            else:
                logging.warning("You are in a DDP environment but no GPU is available, this may cause errors later on")
                n_gpus = 0

            max_memory = {i: max_memory_MB for i in range(n_gpus)}
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            device_map = {"": local_rank}
            max_memory = {"": max_memory[local_rank]} if max_memory_MB is not None else None

        else:
            logging.warning(
                "Using auto device map, we will split the model across GPUs and CPU to fit the model in memory."
            )
            device_map = "auto"
            max_memory = max_memory_MB
    else:
        max_memory = None
        device_map = None
    logging.info(f"We will load the model using the following device map: {device_map} and max_memory: {max_memory}")

    return device_map, max_memory
