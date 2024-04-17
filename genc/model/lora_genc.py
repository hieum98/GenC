from contextlib import contextmanager
import inspect
from itertools import chain
import os
import re
from typing import Any, List, Union
from git import Optional
import torch
import torch.nn as nn
import tqdm
from transformers import PreTrainedModel
from peft import LoraConfig
from peft.utils import infer_device
from accelerate.hooks import AlignDevicesHook
from accelerate.utils import named_module_tensors, offload_state_dict

from genc.model.layer import LoraLayer, dispatch_default
from genc.model.model_utils import _get_submodules, check_target_module_exists


class LoRaGenc(torch.nn.Module):
    def __init__(
            self, 
            model: PreTrainedModel, 
            lora_config: LoraConfig, 
            adapter_name: str = "default"
            ) -> None:
        super().__init__()
        self.model = model

        assert lora_config.target_modules is not None, "You need to specify the target modules for LoRA"
        self.target_module_names = lora_config.target_modules
        if hasattr(self, 'lora_config'):
            modules_to_train = lora_config.modules_to_save
            if any([set(modules_to_train) != set(config.modules_to_save) for config in self.lora_config.values()]):
                raise ValueError("All adapters must have the same modules_to_save. Currently we do not support different modules_to_save for different adapters.")
            self.lora_config[adapter_name] = lora_config
        else:
            self.lora_config = {adapter_name: lora_config}
        self.activate_adpater = adapter_name
        self.inject_adapter(self.model, adapter_name)

        # the `pretraining_tp` is set for some models to simulate Tensor Parallelism during inference to avoid
        # numerical differences, https://github.com/pytorch/pytorch/issues/76232 - to avoid any unexpected
        # behavior we disable that in this line.
        if hasattr(self.model, "config") and hasattr(self.model.config, "pretraining_tp"):
            self.model.config.pretraining_tp = 1
    
    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        new_module = dispatch_default(
            target=target,
            adapter_name=adapter_name,
            lora_config=lora_config,
            **kwargs
        )
        if new_module is None:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `transformers.pytorch_utils.Conv1D`."
            )
        return new_module
    
    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if ('lora_' in name) or ("ranknum" in name):
                weight = child.qweight if hasattr(child, "qweight") else child.weight
                module.to(weight.device)
        
    def _create_and_replace(
        self,
        lora_config: LoraConfig,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
        r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)
        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
        }

        if isinstance(target, LoraLayer):
            target.update_layer(
                adapter_name=adapter_name,
                r=r,
                lora_alpha=alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
            )
        else:
            # Replace the target module with a LoraLayer
            new_module = self._create_new_module(
                lora_config=lora_config,
                adapter_name=adapter_name,
                target=target,
                **kwargs
            )
            self._replace_module(parent, target_name, new_module, target)
        
    def inject_adapter(self, model: torch.nn.Module, adapter_name: str):
        peft_config = self.lora_config[adapter_name]
        self._check_new_adapter_config(peft_config)

        is_target_modules_in_base_model = False
        key_list = [key for key, _ in model.named_modules()]

        model_config = getattr(model, "config", {"model_type": "custom"})
        if hasattr(model_config, "to_dict"):
            model_config = model_config.to_dict()
        
        for key in key_list:
            if not check_target_module_exists(peft_config, key):
                continue
            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(model, key)
            self._create_and_replace(peft_config, adapter_name, target, target_name, parent, current_key=key)
        
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )
        
        self.mark_adapters_as_trainable(model) # Mark only the adapter parameters as trainable

    @staticmethod
    def mark_adapters_as_trainable(model: nn.Module):
        for n, p in model.named_parameters():
            if 'lora_' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False   
    
    def _check_new_adapter_config(self, config: LoraConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.
        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.
        """
        # TODO: there should be a check if any of the existing adapters actually has bias != "none", or else the check
        # does not fully correspond to the error message.
        if (len(self.lora_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)
    
    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def merge_adapter(self, adapter_names: Optional[List[str]] = None):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                with onload_layer(module):
                    module.merge(adapter_names)
    
    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> torch.nn.Module:
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )
    
    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        key_list = [key for key, _ in self.model.named_modules() if 'lora_' not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm.tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "base_layer"):
                    if merge:
                        target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    self._replace_module(parent, target_name, target.get_base_layer(), target)

        return self.model

    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        r"""
        Returns the number of trainable parameters and the number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                if hasattr(param, "element_size"):
                    num_bytes = param.element_size()
                elif not hasattr(param, "quant_storage"):
                    num_bytes = 1
                else:
                    num_bytes = param.quant_storage.itemsize
                num_params = num_params * 2 * num_bytes

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
    
    def print_trainable_parameters(self) -> None:
        """
        Prints the number of trainable parameters in the model.

        Note: print_trainable_parameters() uses get_nb_trainable_parameters() which is different from
        num_parameters(only_trainable=True) from huggingface/transformers. get_nb_trainable_parameters() returns
        (trainable parameters, all parameters) of the Peft Model which includes modified backbone transformer model.
        For techniques like LoRA, the backbone transformer model is modified in place with LoRA modules. However, for
        prompt tuning, the backbone transformer model is unmodified. num_parameters(only_trainable=True) returns number
        of trainable parameters of the backbone transformer model which can be different.
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()

        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
        )
    
    def set_adapter(self, adapter_name: str):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.set_adapter(adapter_name)
        self.activate_adpater = adapter_name
        # Mark additonal modules to train
        orther_modules_to_train = set(self.lora_config[adapter_name].modules_to_save)
        for name, param in self.model.named_parameters():
            if any([n in name for n in orther_modules_to_train]):
                param.requires_grad = True

    def add_adapter(self, adapter_name: str, peft_config: LoraConfig):
        try:
            self.lora_config[adapter_name] = peft_config
            self.inject_adapter(self.model, adapter_name)
        except Exception:
            if adapter_name in self.lora_config:
                del self.lora_config[adapter_name]
            raise
        
@contextmanager
def onload_layer(layer):
    r"""
    A utility for modifying a module containing one or more tuners and a base layer, any of which are offloaded to the
    CPU or disk. Moves a module's sub-modules to the execution device before some action is performed, after that the
    base layer state dictionary is re-assigned (if that layer was offloaded to the disk) and finally the parameters are
    offloaded.

    If the module has no offloaded sub-modules, this function does nothing.

    Args:
        layer ('torch.nn.Module'):
            layer with tuners to be merged
    """

    offloaded_modules = []
    for name, module in layer.named_modules():
        if name in ["", "base_layer"]:
            continue
        if hasattr(module, "_hf_hook") and isinstance(module._hf_hook, AlignDevicesHook) and module._hf_hook.offload:
            module._hf_hook.pre_forward(module)
            offloaded_modules.append(module)

    base_layer_offload = False
    if hasattr(layer, "base_layer") and (
        hasattr(layer.base_layer, "_hf_hook")
        and isinstance(layer.base_layer._hf_hook, AlignDevicesHook)
        and layer.base_layer._hf_hook.offload
    ):
        # check if the base layer is disk-offloaded (must contain a 'dataset' and an offload index)
        if torch.device("meta") in layer.base_layer._hf_hook.original_devices.values() and hasattr(
            layer.base_layer._hf_hook.weights_map, "dataset"
        ):
            # find the disk-offload index (maps modules to safetensors) from the `dataset` (OffloadedWeightsLoader object)
            index = layer.base_layer._hf_hook.weights_map.dataset.index
            module_name = list(dict(layer.base_layer._hf_hook.weights_map.dataset).keys())[0]  # any module will do
            file_name = index[module_name]["safetensors_file"]
            base_name_arr = []
            # get effective dir name
            for i in os.path.split(file_name):
                if "--" in i:
                    base_name_arr.append(i)
                    break
                base_name_arr.append(i)
            base_name = os.path.join(*base_name_arr)
            safetensors_filename = base_name + "-merged"
        layer.base_layer._hf_hook.pre_forward(layer.base_layer)
        base_layer_offload = True

    yield

    for module in offloaded_modules:
        module._hf_hook.post_forward(module, torch.tensor([]))

    if base_layer_offload:
        # re-make weights map (must be on cpu to send params to the disk via memmap if disk offload)
        layer.base_layer._hf_hook.weights_map = {
            name: param.to("cpu") for name, param in named_module_tensors(layer.base_layer)
        }
        # offload weights map to disk if original device is the disk
        if torch.device("meta") in layer.base_layer._hf_hook.original_devices.values() and hasattr(
            layer.base_layer._hf_hook.weights_map, "dataset"
        ):
            # rewrite directory with merged weights
            offload_state_dict(safetensors_filename, layer.base_layer._hf_hook.weights_map)
        layer.base_layer._hf_hook.post_forward(layer.base_layer, torch.tensor([]))
