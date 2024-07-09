from typing import Dict, List, Optional, Tuple, Union, cast
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, BatchEncoding

from genc.model.genc import LlamaEmbeddingLM, MistralEmbeddingLM, PhiEmbeddingLM
from genc.special_tokens import SPECILA_TOKENS
from genc.trainer.trainer_utils import chunked_cross_entropy, get_batch_logps


class GenCLM(torch.nn.Module):
    def __init__(
            self,
            model_weights_name_or_path: str,
            is_old: bool = False,
            pretrained_type: str = 'mistral',
            use_bidirectional: bool = True,
            normalized: bool = True,
            pooling_method: str = "mean",
            torch_dtype: torch.dtype = torch.bfloat16,
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
        if pretrained_type=="llama":
            model_class = LlamaEmbeddingLM
        elif pretrained_type=="mistral":
            model_class = MistralEmbeddingLM
        elif pretrained_type=="phi":
            model_class = PhiEmbeddingLM
        else:
            raise ValueError(f"Model type not recognized: {model_weights_name_or_path}")
        
        print(f"Loading model from {model_weights_name_or_path}")
        self.model = model_class.from_pretrained(
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
        if is_old:
            base_bos: str = "<s>"
            user_bos: str = "<|user|>\n"
            user_eos: str = ""
            embed_bos: str = "\n<|embed|>\n"
            embed_eos: str = "</e>"
            assistant_bos = "\n<|assistant|>\n"
            assistant_eos = "</s>"
            self.emb_prompt_format = base_bos + user_bos + "{prompt}" + user_eos + embed_bos
            self.emb_example_format = self.emb_prompt_format + "{example}" + embed_eos
            self.gen_prompt_format = base_bos + user_bos + "{prompt}" + user_eos + assistant_bos
            self.gen_example_format = self.gen_prompt_format + "{response}" + assistant_eos
        else:
            special_tokens = SPECILA_TOKENS[pretrained_type]
            bos = special_tokens.get("bos", "")
            user_bos = special_tokens.get("user_bos", "")
            eos = special_tokens.get("eos", "")
            eot = special_tokens.get("eot", "")
            assistant_bos = special_tokens.get("assistant_bos", "")
            self.emb_prompt_format = bos + user_bos + "{prompt}" + "\n"
            self.emb_example_format = self.emb_prompt_format + "{example}" + eot + eos
            self.gen_prompt_format = bos + user_bos + "{prompt}" + eot + assistant_bos
            self.gen_example_format = self.gen_prompt_format + "{response}" + eot + eos
        self.label_pad_token_id = -100
        
        print(f"The text prompt format is: \n{self.emb_prompt_format}")
        print(f"The text example format is: \n{self.emb_example_format}")
        print(f"The generation prompt format is: \n{self.gen_prompt_format}")
        print(f"The generation example format is: \n{self.gen_example_format}")

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

class GenCLMRetrieval(GenCLM):
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


class GenCLMReranker(GenCLM):
    def tokenize_example_for_reranking(
            self, 
            example: Tuple[str, str, str], # Tuple of [instruction, query, candidate]
            max_length: int = 512,
            ) -> BatchEncoding:
        prompt = f"{example[0]}\n{example[1]}"
        prompt = self.gen_prompt_format.format(prompt=prompt)
        example = self.gen_example_format.format(prompt=prompt, response=example[2])
        model_inputs = self.tokenizer(
            text=example,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=False, # already added
        )

        # Add labels to the model inputs
        model_inputs["labels"] = model_inputs["input_ids"].copy()

        prompt_ids = self.tokenizer(
            prompt,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
            )["input_ids"]
        if len(prompt_ids) > len(model_inputs["input_ids"]):
            raise ValueError(f"Prompt is longer than the model input\n Prompt: {prompt}\n Response: {example}")
        prompt_len = len(prompt_ids)
        loss_weight_mask = np.ones(len(model_inputs["labels"]), dtype=np.float32)
        loss_weight_mask[0: prompt_len] = 0.0
        model_inputs["loss_weight_mask"] = loss_weight_mask
        return model_inputs
    
    def pad_gen_example(self, generative: List[BatchEncoding]):
        labels = [item['labels'] for item in generative] if 'labels' in generative[0] else None
        loss_weight_mask = [item['loss_weight_mask'] for item in generative] if 'loss_weight_mask' in generative[0] else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            max_label_length = (
                (max_label_length + 8 - 1)
                // 8
                * 8
            )  # padding to multiple of 8
            padding_side = self.tokenizer.padding_side
            for item in generative:
                remainder = [self.label_pad_token_id] * (max_label_length - len(item["labels"]))
                if isinstance(item["labels"], list):
                    item["labels"] = (
                        item["labels"] + remainder if padding_side == "right" else remainder + item["labels"]
                    )
                elif padding_side == "right":
                    item["labels"] = np.concatenate([item["labels"], remainder]).astype(np.int64)
                else:
                    item["labels"] = np.concatenate([remainder, item["labels"]]).astype(np.int64)

        if loss_weight_mask is not None:
            max_label_length = max(len(l) for l in loss_weight_mask)
            max_label_length = (
                (max_label_length + 8 - 1)
                // 8
                * 8
            ) # padding to multiple of 8
            padding_side = self.tokenizer.padding_side
            for item in generative:
                remainder = [0.0] * (max_label_length - len(item["loss_weight_mask"]))
                if isinstance(item["loss_weight_mask"], list):
                    item["loss_weight_mask"] = (
                        item["loss_weight_mask"] + remainder if padding_side == "right" else remainder + item["loss_weight_mask"]
                    )
                elif padding_side == "right":
                    item["loss_weight_mask"] = np.concatenate([item["loss_weight_mask"], remainder]).astype(np.float32)
                else:
                    item["loss_weight_mask"] = np.concatenate([remainder, item["loss_weight_mask"]]).astype(np.float32)
        generative = self.tokenizer.pad(
            generative, 
            pad_to_multiple_of=8,
            return_tensors="pt"
            )
        return generative
    
    def compute_gen_loss(
            self,
            labels, 
            logits,
            loss_weight_mask=None,
            ):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_weight_mask = loss_weight_mask[..., 1:].contiguous() if loss_weight_mask is not None else None
        loss = chunked_cross_entropy(
            shift_logits, shift_labels, loss_weight_mask=loss_weight_mask, ignore_index=-100
        )
        return loss
    
    @torch.no_grad()
    def predict(
        self,
        sentences: List[List[str]], # List of [query, candidate]
        batch_size: int = 256,
        max_length: int = 2048,
        instruction: str = "",
        **kwargs,
        ) -> np.ndarray:
        """Used for predicting the scores of retrieval tasks"""
        if self.num_gpus > 1:
            batch_size *= self.num_gpus

        input_was_string = False
        if len(sentences) == 2 and isinstance(sentences[0], str):
            sentences = [sentences]
            input_was_string = True
        
        pred_scores = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Batches", disable=len(sentences)<256):
            sentence_batch = [[instruction, s[0], s[1]] for s in sentences[start_index:start_index+batch_size]]
            inputs = [self.tokenize_example_for_reranking(example, max_length) for example in sentence_batch]
            inputs = self.pad_gen_example(inputs)
            labels = inputs["labels"]
            loss_weight_mask = inputs["loss_weight_mask"]
            inputs = {
                "input_ids": inputs["input_ids"].to(self.device),
                "attention_mask": inputs["attention_mask"].to(self.device),
                "is_gen": True
            }
            logits = self.model(**inputs)['logits'].detach().cpu() # (batch_size, seq_len, vocab_size)
            all_losses = []
            for i in range(logits.size(0)):
                loss = self.compute_gen_loss(
                    labels=labels[i],
                    logits=logits[i],
                    loss_weight_mask=loss_weight_mask[i],
                )
                all_losses.append(loss)
            all_losses = torch.stack(all_losses, dim=0)
            score = 1.0 / torch.exp(all_losses)
            pred_scores.append(score)
        pred_scores = torch.cat(pred_scores, dim=0).numpy() # (num_samples, 1)
        if input_was_string:
            pred_scores = pred_scores[0]

        return pred_scores
    
    def rank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        batch_size: int = 256,
        max_length: int = 512,
        instruction: str = "",
        **kwargs,
        ):
        """Used for ranking documents based on a query"""
        query_doc_pairs = [[query, doc] for doc in documents]
        scores = self.predict(
            sentences=query_doc_pairs,
            batch_size=batch_size,
            max_length=max_length,
            instruction=instruction,
        )
        results = []
        for i in range(len(scores)):
            results.append({"corpus_id": i, "score": scores[i]})
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_k]
