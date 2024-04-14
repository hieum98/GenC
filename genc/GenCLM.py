from typing import Dict, List, Tuple, Union, cast
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, BatchEncoding

from genc.model.genc import MistralEmbeddingLM


class GenCLM(torch.nn.Module):
    def __init__(
            self,
            model_weights_name_or_path: str,
            use_bidirectional: bool = False,
            normalized: bool = True,
            pooling_method: str = "mean",
            torch_dtype: torch.dtype = torch.bfloat16,
            base_bos: str = "<s>",
            user_bos: str = "<|user|>\n",
            user_eos: str = "",
            embed_bos: str = "\n<|embed|>\n",
            embed_eos: str = "</e>",
            is_inference: bool = True,
            **kwargs,
            ) -> None:
        super().__init__()

        config = AutoConfig.from_pretrained(
            model_weights_name_or_path,
            trust_remote_code=True,
            use_cache=False
        )
        model_args = [use_bidirectional, normalized, pooling_method]
        self.model = MistralEmbeddingLM.from_pretrained(
            model_weights_name_or_path,
            *model_args,
            config=config,
            torch_dtype=torch_dtype,
            **kwargs,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_weights_name_or_path,
            padding_side="right", # Has to be right so masking of instruction tokens works correctly
            trust_remote_code=True,
        )
        self.emb_prompt_format = base_bos + user_bos + "{prompt}" + user_eos + embed_bos
        self.emb_example_format = self.emb_prompt_format + "{example}" + embed_eos

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.num_gpus = torch.cuda.device_count()
        self.num_gpus = 1
        
        print(f"Created GenCLM: {self.model.dtype} dtype, {pooling_method} pool, normalized {normalized}, use_bidirectional {use_bidirectional}")

        if is_inference:
            self.model.eval()
            if not("device_map" in kwargs):
                self.model.to(self.device)
                # Parallelize embedding model
                self.num_gpus = torch.cuda.device_count()
                if self.num_gpus > 1:
                    print(f"----------Using {self.num_gpus} data-parallel GPUs----------")
                    self.model = torch.nn.DataParallel(self.model)

    def tokenize_example(
            self, 
            example: Tuple[str, str],
            max_length: int = 512,
            ) -> BatchEncoding:
        emb_prompt = self.emb_prompt_format.format(prompt=example[0])
        emb_example = self.emb_example_format.format(prompt=example[0], example=example[1])
        model_inputs = self.tokenizer(
            text=emb_example,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=False, # already added
        )
        
        # Find the prompt length
        prompt_ids = self.tokenizer(
            emb_prompt,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
            )["input_ids"]
        if len(prompt_ids) > len(model_inputs["input_ids"]):
            raise ValueError("Prompt is longer than the model input")
        model_inputs["prompt_length"] = len(prompt_ids)
        return model_inputs
    
    @torch.no_grad()
    def encode(
        self,
        sentences: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
        instruction: str = "",
        **kwargs,
        ):
        if self.num_gpus > 1:
            batch_size *= self.num_gpus

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True
        
        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Batches", disable=len(sentences)<256):
            sentence_batch = [[instruction, s] for s in sentences[start_index:start_index+batch_size]]
            inputs = [self.tokenize_example(example, max_length) for example in sentence_batch]
            inputs = self.tokenizer.pad(inputs, pad_to_multiple_of=8, return_tensors="pt")
            inputs = {
                "input_ids": inputs["input_ids"].to(self.device),
                "attention_mask": inputs["attention_mask"].to(self.device),
                "prompt_length": inputs["prompt_length"],
            }
            embeddings = self.model.encode(**inputs) # (batch_size, hidden_size)
            all_embeddings.append(embeddings.cpu().to(torch.float32).numpy())
        
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        if input_was_string:
            all_embeddings = all_embeddings[0]
        
        return all_embeddings
    
    def encode_queries(self, queries: Union[List[str], str], **kwargs) -> np.ndarray:
        """Used for encoding the queries of retrieval or reranking tasks"""
        return self.encode(queries, **kwargs)

    def encode_corpus(self, corpus: Union[List[str], str, List[Dict[str, str]]], **kwargs) -> np.ndarray:
        """Used for encoding the corpus of retrieval tasks"""
        if isinstance(corpus, dict):
            corpus = [corpus]
        if isinstance(corpus, list) and isinstance(corpus[0], dict):
            corpus = [
                doc["title"] + " " + doc["text"] if "title" in doc 
                else doc["text"] for doc in corpus
            ]
        return self.encode(corpus, **kwargs)



