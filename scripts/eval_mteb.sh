#!/bin/bash
#SBATCH --job-name=mteb
#SBATCH --account=uonlp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          
#SBATCH --mem=100G
#SBATCH --constraint=gpu-80gb,no-mig
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --cpus-per-task=20
#SBATCH --output=/home/hieum/uonlp/LLM_Emb/mistral-7b-dpoc-msmarco-%j.out
#SBATCH --error=/home/hieum/uonlp/LLM_Emb/mistral-7b-dpoc-msmarco-%j.err
#SBATCH --exclusive
#SBATCH --array=0-68%69

# FEWSHOT: 0-9%10

######################
### Set enviroment ###
######################
# Activate conda environment
source activate llm
cd /home/hieum/uonlp/LLM_Emb
export WANDB_PROJECT="mistral-7b-dpoc-msmarco"
######################

######################
#### Set network #####
######################

######################

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

# DS=${ALLDS[$SLURM_ARRAY_TASK_ID]}

export TRANSFORMERS_CACHE=/home/hieum/uonlp/hf_cache
export HF_HOME=/home/hieum/uonlp/hf_cache

# For each dataset in ALLDS run the evaluation script
for DS in ${ALLDS[@]}; do
    echo "Running evaluation for MTEB"
    python -m eval.eval_mteb \
    --model_name_or_path checkpoint/7b-esft_msmarco-50 \
    --attn_implementation sdpa \
    --use_bidirectional \
    --task_names $DS \
    --instruction_set medi2 \
    --instruction_format genclm \
    --batch_size 64 \
    --pipeline_parallel \
    --pooling_method mean
done

