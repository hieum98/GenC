#!/bin/bash
#SBATCH --job-name=ds
#SBATCH --account=uonlp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          
#SBATCH --mem=100G
#SBATCH --partition=preempt
#SBATCH --constraint=gpu-40gb|gpu-80gb|h100|gpu-10gb
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --cpus-per-task=10
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/hieum/uonlp/LLM_Emb/hard-%j.out
#SBATCH --error=/home/hieum/uonlp/LLM_Emb/hard-%j.err
#SBATCH --array=0-21


######################
### Set enviroment ###
######################
# Activate conda environment
source activate llm
cd /home/hieum/uonlp/LLM_Emb
######################

DATA=(
    'scifact'
    'stackexchange'
    'ccnews_title_text'
    'yahoo_answers_title_answer'
    'fever'
    'eli5_question_answer'
    'reddit-title-body'
    'xsum'
    'codesearchnet'
    'trivia_qa'
    'S2ORC_title_abstract'
    'hotpot_qa'
    'squad'
    'pubmedqa'
    'fiqa'
    'nq'
    'specter_train_triples'
    'allnli'
    'msmarco_document'
    'quora_duplicates'
    'msmarco_passage'
)

DS=${DATA[$SLURM_ARRAY_TASK_ID]}
echo "Processing $DS"

python scripts/process_hard_genclm.py --dataset $DS

