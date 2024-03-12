import json
import logging
import os
from pathlib import Path 
import torch 
import torch.distributed as dist
from transformers import HfArgumentParser, set_seed

from hdlm.model.load_model import load_model, BASE_BOS, TURN_SEP, USER_BOS, USER_EOS, EMBED_BOS, EMBED_EOS, ASSISTANT_BOS, ASSISTANT_EOS
from hdlm.dataset.load_data import load_data_for_sft
from hdlm.dataset.data import CustomDatasetForSFT, CustomCollatorForSFT, CustomRandomSampler
from hdlm.trainers.sft_trainer import SFTTrainer
from hdlm.arguments import ModelArguments, DataArguments, CustomTrainingArguments

logger = logging.getLogger(__name__)


def get_correct_torch_dtype(
    quantization: int,
    model_args: ModelArguments,
    training_args: CustomTrainingArguments,
) -> "str":
    """
    Returns the correct torch dtype based on the model and training arguments (if quantization is enabled).

    Args:
        quantization (`int`, optional):
            '4' or '8' for 4 bits or 8 bits quantization or None for 16/32bits training. Defaults to `None`.
        model_args (:class:`~transformers.ModelArguments`):
            The model arguments.
        training_args (:class:`~transformers.Seq2SeqTrainingArguments`):
            The training arguments.

    Returns:
        :obj:`str`: The correct torch dtype.
    """

    if isinstance(quantization, str):
        quantization = int(quantization)

    if quantization in [4, 8]:
        if training_args.fp16:
            if model_args.torch_dtype in ["auto", None]:
                logging.warning(
                    "Quantification and fp16 are enabled, but torch_dtype is not set. Setting torch_dtype to float16."
                )

            elif model_args.torch_dtype != "float16":
                logging.warning(
                    f"Quantification and fp16 are enabled, but torch_dtype is set to {model_args.torch_dtype}. "
                    "This can cause issues. We will override torch_dtype to float16."
                )
            return "float16"

        elif training_args.bf16:
            if model_args.torch_dtype in ["auto", None]:
                logging.warning(
                    "Quantification and bf16 are enabled, but torch_dtype is not set. Setting torch_dtype to bfloat16."
                )
            elif model_args.torch_dtype != "bfloat16":
                logging.warning(
                    f"Quantification and bf16 are enabled, but torch_dtype is set to {model_args.torch_dtype}. "
                    "This can cause issues. We will override torch_dtype to bfloat16."
                )
            return "bfloat16"

    return model_args.torch_dtype

def train_sft(
        model_args: ModelArguments, 
        data_args: DataArguments, 
        training_args: CustomTrainingArguments
        ):
    # Sanity check
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to bypass."
        )
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    # Load model and tokenizer
    logger.info(f"Loading {model_args.model_name_or_path} model...")
    model, tokenizer, config = load_model(
        model_weights_name_or_path=model_args.model_name_or_path,
        normalized=model_args.model_name_or_path,
        loss_gen_factor=training_args.loss_gen_factor,
        pooling_method=model_args.pooling_method,
        loss_gen_type=training_args.loss_gen_type,
        quantization=training_args.quantization,
        use_gradient_checkpointing=training_args.gradient_checkpointing,
        use_lora=model_args.use_lora,
        lora_weights_name_or_path=model_args.lora_weights_name_or_path,
        lora_target_modules=model_args.lora_target_modules,
        lora_r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        torch_dtype=get_correct_torch_dtype(
            quantization=training_args.quantization, model_args=model_args, training_args=training_args
        ),
        inference=False,
        fsdp=len(training_args.fsdp) > 1 or training_args.fsdp_config is not None,
    )

    # Handle grad accumulation manually inside forward if use Gradcache.
    if training_args.use_gc:
        training_args.per_device_train_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        training_args.gradient_accumulation_steps = 1
        logger.info("Using GradCache with mini batch size %d and full batch size %d", training_args.gc_mini_batch_size, training_args.per_device_train_batch_size)
    
    # Load data
    emb_train_ds, gen_train_ds, ds_name_to_samples, ds_embedding_lens = load_data_for_sft(data_args=data_args, tokenizer=tokenizer)
    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(os.path.join(training_args.output_dir, "dataset_num_samples.json"), "w") as f:
        json.dump(ds_name_to_samples, f)

    dataset = CustomDatasetForSFT(
        dataset=[emb_train_ds, gen_train_ds],
        args=data_args,
        mode=training_args.mode,
        max_seq_len=512,
    )
    data_collator = CustomCollatorForSFT(
        tokenizer=tokenizer,
        passage_max_len=data_args.passage_max_len,
        generative_max_len=data_args.generative_max_len,
        base_bos=BASE_BOS,
        turn_sep=TURN_SEP,
        user_bos=USER_BOS,
        user_eos=USER_EOS,
        embed_bos=EMBED_BOS,
        embed_eos=EMBED_EOS,
        assistant_bos=ASSISTANT_BOS,
        assistant_eos=ASSISTANT_EOS,
        prefixlm=data_args.prefixlm,
        prompt_loss_weight=data_args.prompt_loss_weight,
        padding=True,
        label_pad_token_id=-100
    )

    # Initial Trainer
    if training_args.use_gc:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
            tokenizer=tokenizer,
            gc_mini_batch_size=training_args.gc_mini_batch_size,
            use_gc=True,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )

    if len(ds_embedding_lens) > 1 and dataset.total_emb_size >= dataset.total_gen_size:
        assert training_args.dataloader_drop_last, "Multiple datasets are only supported with dropping the last incomplete batch, set `--dataloader_drop_last`"
        logger.info("Embedding dataset lengths: %s", ds_embedding_lens)
        # Multiple embedding datasets & we want to make sure each batch mostly comes from one dataset
        # Set custom sampler, see https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/trainer.py#L785
        total_bs = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        total_bs = total_bs * dist.get_world_size() if dist.is_initialized() else total_bs
        trainer._get_train_sampler = lambda: CustomRandomSampler(
            total_batch_size=total_bs, ds_lens=ds_embedding_lens
            )

    if training_args.mode == "unified":
        # Track all losses
        from transformers.integrations import WandbCallback
        from transformers.integrations.integration_utils import rewrite_logs
        from transformers.trainer_pt_utils import distributed_concat
        class WandbCustomCallback(WandbCallback):
            def on_log(self, args, state, control, model=None, logs=None, **kwargs):
                if self._wandb is None: return
                if not self._initialized: self.setup(args, state, model)
                if hasattr(state, "loss_emb") and hasattr(state, "loss_gen"):
                    # Gather & avg across gpus like for actual loss
                    # https://github.com/huggingface/transformers/blob/bc72b4e2cdcbc80d5f56731f35dbc9c18b4c8de6/src/transformers/trainer.py#L2257
                    if (args.distributed_state is not None and args.distributed_state.distributed_type != "NO") or (
                        args.distributed_state is None and args.local_rank != -1):
                        state.loss_emb = distributed_concat(state.loss_emb).mean().item()
                        state.loss_gen = distributed_concat(state.loss_gen).mean().item()
                    else:
                        state.loss_emb = state.loss_emb.mean().item()
                        state.loss_gen = state.loss_gen.mean().item()
                    if state.is_world_process_zero:
                        self._wandb.log({
                            **rewrite_logs(logs),
                            "train/global_step": state.global_step,
                            "train/loss_emb": state.loss_emb,
                            "train/loss_gen": state.loss_gen,
                        })
                    del state.loss_emb
                    del state.loss_gen
                else:
                    if state.is_world_process_zero:
                        self._wandb.log({
                            **rewrite_logs(logs),
                            "train/global_step": state.global_step,
                        })

        trainer.add_callback(WandbCustomCallback())
    
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    logger.info("Starting training")
    trainer.train()
    
    # The below does not save if state dict type is `SHARDED_STATE_DICT`
    trainer.save_model()

    # To be safe do another FS save
    if (trainer.is_fsdp_enabled) and (trainer.accelerator.state.fsdp_plugin.state_dict_type != "FULL_STATE_DICT"):
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        fsd_path = os.path.join(training_args.output_dir, "full_state_dict")
        os.makedirs(fsd_path, exist_ok=True)
        trainer.save_model(fsd_path)

    # Save tokenizer & config for easy usage afterwards
    if trainer.is_world_process_zero(): 
        tokenizer.save_pretrained(training_args.output_dir)
        config.to_json_file(training_args.output_dir + "/config.json")



if __name__=='__main__':
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments = args[0]
    data_args: DataArguments = args[1]
    training_args: CustomTrainingArguments = args[2]

    train_sft(model_args, data_args, training_args)



