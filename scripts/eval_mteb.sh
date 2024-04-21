#!/bin/bash
#SBATCH --job-name=mteb
#SBATCH --account=uonlp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          
#SBATCH --mem=100G
#SBATCH --constraint=gpu-40gb|gpu-80gb|gpu-20gb
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --cpus-per-task=10
#SBATCH --output=/home/hieum/uonlp/LLM_Emb/mteb-%j.out
#SBATCH --error=/home/hieum/uonlp/LLM_Emb/mteb-%j.err
#SBATCH --array=0-12

# FEWSHOT: 0-9%10
# ALLDS: 0-69

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
AmazonCounterfactualClassification
AmazonPolarityClassification
AmazonReviewsClassification
ArguAna
ArxivClusteringP2P
ArxivClusteringS2S
AskUbuntuDupQuestions
BIOSSES
Banking77Classification
BiorxivClusteringP2P
BiorxivClusteringS2S
CQADupstackAndroidRetrieval
CQADupstackEnglishRetrieval
CQADupstackGamingRetrieval
CQADupstackGisRetrieval
CQADupstackMathematicaRetrieval
CQADupstackPhysicsRetrieval
CQADupstackProgrammersRetrieval
CQADupstackStatsRetrieval
CQADupstackTexRetrieval
CQADupstackUnixRetrieval
CQADupstackWebmastersRetrieval
CQADupstackWordpressRetrieval
ClimateFEVER
DBPedia
EmotionClassification
FEVER
FiQA2018
HotpotQA
ImdbClassification
MSMARCO
MTOPDomainClassification
MTOPIntentClassification
MassiveIntentClassification
MassiveScenarioClassification
MedrxivClusteringP2P
MedrxivClusteringS2S
MindSmallReranking
NFCorpus
NQ
QuoraRetrieval
RedditClustering
RedditClusteringP2P
SCIDOCS
SICK-R
STS12
STS13
STS14
STS15
STS16
STS17
STS22
STSBenchmark
SciDocsRR
SciFact
SprintDuplicateQuestions
StackExchangeClustering
StackExchangeClusteringP2P
StackOverflowDupQuestions
SummEval
TRECCOVID
Touche2020
ToxicConversationsClassification
TweetSentimentExtractionClassification
TwentyNewsgroupsClustering
TwitterSemEval2015
TwitterURLCorpus
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

DS=${QUICK_EVAL[$SLURM_ARRAY_TASK_ID]}

export TRANSFORMERS_CACHE=/home/hieum/uonlp/hf_cache
export HF_HOME=/home/hieum/uonlp/hf_cache

# For each dataset in ALLDS run the evaluation script
echo "Running evaluation for MTEB $DS"
python -m eval.eval_mteb \
--model_name_or_path checkpoint/7b-esft_simcse-100 \
--attn_implementation sdpa \
--use_bidirectional \
--task_names $DS \
--instruction_set medi2 \
--instruction_format genclm \
--batch_size 64 \
--pipeline_parallel \
--pooling_method mean

