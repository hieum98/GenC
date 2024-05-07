#!/bin/bash
#SBATCH --job-name=mteb
#SBATCH --account=uonlp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          
#SBATCH --mem=150G
#SBATCH --constraint=h100|gpu-80gb|gpu-40gb
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/hieum/uonlp/LLM_Emb/mteb-%j.out
#SBATCH --error=/home/hieum/uonlp/LLM_Emb/mteb-%j.err
#SBATCH --array=0-55

# Quick eval: 0-11
# FEWSHOT: 0-9
# ALLDS: 0-55

######################
### Set enviroment ###
######################
# Activate conda environment
source activate llm
cd /home/hieum/uonlp/LLM_Emb
export WANDB_PROJECT="mteb"
######################

######################
#### Set network #####
######################

######################
QUICK_EVAL=(
    # Classification
    Banking77Classification
    EmotionClassification
    # Clustering
    MedrxivClusteringS2S
    # PairClassification
    TwitterSemEval2015
    # Reranking
    AskUbuntuDupQuestions
    # Retrieval
    ArguAna
    NFCorpus
    SciFact
    # STS
    BIOSSES
    STS17
    STSBenchmark
    # Summarization
    SummEval
)

ALLDS=(
    "AmazonCounterfactualClassification"
    "AmazonPolarityClassification"
    "AmazonReviewsClassification"
    "Banking77Classification"
    "EmotionClassification"
    "ImdbClassification"
    "MassiveIntentClassification"
    "MassiveScenarioClassification"
    "MTOPDomainClassification"
    "MTOPIntentClassification"
    "ToxicConversationsClassification"
    "TweetSentimentExtractionClassification"

    "ArxivClusteringP2P"
    "ArxivClusteringS2S"
    "BiorxivClusteringP2P"
    "BiorxivClusteringS2S"
    "MedrxivClusteringP2P"
    "MedrxivClusteringS2S"
    "RedditClustering"
    "RedditClusteringP2P"
    "StackExchangeClustering"
    "StackExchangeClusteringP2P"
    "TwentyNewsgroupsClustering"

    "SprintDuplicateQuestions"
    "TwitterSemEval2015"
    "TwitterURLCorpus"

    "AskUbuntuDupQuestions"
    "MindSmallReranking"
    "SciDocsRR"
    "StackOverflowDupQuestions"

    "ArguAna"
    "ClimateFEVER"
    "CQADupstackTexRetrieval"
    "DBPedia"
    "FEVER"
    "FiQA2018"
    "HotpotQA"
    "MSMARCO"
    "NFCorpus"
    "NQ"
    "QuoraRetrieval"
    "SCIDOCS"
    "SciFact"
    "Touche2020"
    "TRECCOVID"

    "BIOSSES"
    "SICK-R"
    "STS12"
    "STS13"
    "STS14"
    "STS15"
    "STS16"
    "STS17"
    "STS22"
    "STSBenchmark"

    "SummEval"
)

# Dataets for fewshot exps
# FEWSHOT=(
# Banking77Classification
# EmotionClassification
# ImdbClassification
# BiorxivClusteringS2S
# SprintDuplicateQuestions
# TwitterSemEval2015
# TwitterURLCorpus
# AskUbuntuDupQuestions
# ArguAna
# SCIDOCS
# STS12
# SummEval
# )

REMAIN=(
    "StackExchangeClusteringP2P"
    "MindSmallReranking"
    "MSMARCO"
)

# DS=${REMAIN[$SLURM_ARRAY_TASK_ID]}
DS=${ALLDS[$SLURM_ARRAY_TASK_ID]}

export TRANSFORMERS_CACHE=/home/hieum/uonlp/hf_cache
export HF_HOME=/home/hieum/uonlp/hf_cache

# For each dataset in ALLDS run the evaluation script
echo "Running evaluation for MTEB on $DS"
python -m eval.eval_mteb \
    --model_name_or_path checkpoint/esft_msmarco_7b_instruct \
    --pretrained_type mistral \
    --attn_implementation sdpa \
    --use_bidirectional \
    --task_names $DS \
    --instruction_set genclm \
    --instruction_format genclm \
    --batch_size 8 \
    --pipeline_parallel \
    --pooling_method mean


rm /home/hieum/uonlp/LLM_Emb/mteb-$SLURM_JOB_ID.out
rm /home/hieum/uonlp/LLM_Emb/mteb-$SLURM_JOB_ID.err


