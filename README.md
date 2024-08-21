# *GenC: Generative Contrastive Learning for Passage Retrieval with Large Language Models*

[![HF Link](https://img.shields.io/badge/HF%20Models-GenC-FFD21E.svg)](https://huggingface.co/Hieuman/GenC-LlaMa)

Generative Contrastive Learning (GenC) is a novel training framework that unlocks the power of LLMs for retrieval tasks by seamlessly integrating their generation abilities into the representation learning process. GenC optimizes passage retrieval by not only leveraging the representation capacities of LLMs to learn dense embeddings but also directly exploiting the models’ generation probabilities as new training signals.

## Installation
To use GenC, install evironment from ```environment.yaml```
```bash
conda env create -f environment.yaml
```

After that, you can install our package from source by
```bash
pip install -e .
```

## Getting started
The GenCLM class enhances HuggingFace models by adding support for bidirectional processing in decoder-only Large Language Models (LLMs), as well as sequence encoding and pooling operations. The steps below showcase an example on how to use this.

### Pretraining the model
Initializing GenC model using pretrained LLMs is straightforward. The `model_weights_name_or_path` argument of GenCLM takes a base model identifier/path. By default, the models are loaded with bidirectional connections enabled. This can be turned off by passing `use_bidirectional=False` to the `from_pretrained` method.

```python
import torch
from genc.GenCLM import GenCLMRetrieval

genclm = GenCLMRetrieval(model_weights_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",)
```
By default the GenCLM model uses the `mean` pooling strategy. You can change the pooling strategy by passing the `pooling_method` argument to the `from_pretrained` method.

### Inference
This model now returns the text embedding for any input in the form of `str` or `List[str]`. The model also can receive instruction alongside the sentence.

```python
# Encoding queries using instructions
instruction =  "Given a web search query, retrieve relevant passages that answer the query:"
queries = [
    "how much protein should a female eat",
    "summit define",
]
q_reps = genclm.encode(sentences=queries)

# Encoding documents. Instruction are not required for documents
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
]
d_reps = genclm.encode(sentences=documents)

# Compute cosine similarity
q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
d_reps_norm = torch.nn.functional.normalize(d_reps, p=2, dim=1)
cos_sim = torch.mm(q_reps_norm, d_reps_norm.transpose(0, 1))

print(cos_sim)
```

## Model List

We publish three pretrained model including [Meta-Llama-3-8B](https://huggingface.co/Hieuman/GenC-LlaMa); [Mistral-2-7B](https://huggingface.co/Hieuman/GenC-Mistral); [Phi-1.5B](https://huggingface.co/Hieuman/GenC-Phi1.5)


## Training 
For training, we use [MSMARCO](https://huggingface.co/datasets/BeIR/msmarco) dataset. To use the training script, the downloaded dataset should be processed by runing:
```bash
python -m scripts.process_msmarco
```

To train the Meta-Llama-3-8B model, run the following command:

```bash
python -m genc.main \
    --config_file scripts/configs/genclm/msmarco_llamma3.yaml \
    --nodes 4 \
    --devices 1 \
    --mode edpo \
    --output_dir output/edpo_llamma3
```

Note that, for all of these steps, you should to change the data path to your correct data path accordingly.


## Evaluation 
To evaluate the model on the MTEB benchmark, run the following command:
```bash
python -m eval.eval_mteb \
    --model_name_or_path output/edpo_msmarco_8b_v2/edpo_msmarco_8b_v2 \
    --pretrained_type llama \
    --attn_implementation sdpa \
    --use_bidirectional \
    --task_names MSMARCO \
    --instruction_set genclm \
    --instruction_format genclm \
    --batch_size 8 \
    --pipeline_parallel \
    --pooling_method mean 
```

The evaluation script supports all our publish models.

## Bugs or questions?
If you have any questions about the code, feel free to open an issue on the GitHub repository.

