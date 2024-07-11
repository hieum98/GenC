import argparse
import os
from functools import partial
from mteb import MTEB
import torch

from  genc.GenCLM import GenCLM, GenCLMReranker, GenCLMRetrieval


SET_TO_TASK_TO_DS_TO_PROMPT = {
    'genclm': {
        "Classification": {
            'Banking77Classification': 'Given a online banking query, find the corresponding intents',
            'EmotionClassification': 'Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise',
            'AmazonCounterfactualClassification': 'Classify a given Amazon customer review text as either counterfactual or not-counterfactual',
            'ImdbClassification': 'Classify the sentiment expressed in the given movie review text from the IMDB dataset',
            'MassiveIntentClassification': 'Given a user utterance as query, find the user intents',
            'MassiveScenarioClassification': 'Given a user utterance as query, find the user scenarios',
            'MTOPDomainClassification': 'Classify the intent domain of the given utterance in task-oriented conversation',
            'MTOPIntentClassification': 'Classify the intent of the given utterance in task-oriented conversation',
            'ToxicConversationsClassification': 'Classify the given comments as either toxic or not toxic',
            'AmazonPolarityClassification': 'Classify Amazon reviews into positive or negative sentiment',
            'AmazonReviewsClassification': 'Classify the given Amazon review into its appropriate rating category',
            'TweetSentimentExtractionClassification': 'Classify the sentiment of a given tweet as either positive, negative, or neutral',
        },
        "Clustering": {
            'ArxivClusteringP2P': 'Represent the title and abstract of a paper to identify its category',
            'ArxivClusteringS2S': 'Represent the title of a paper to identify its category',
            'BiorxivClusteringP2P': 'Represent the title and abstract of a paper to identify its category',
            'BiorxivClusteringS2S': 'Represent the title of a paper to identify its category',
            'MedrxivClusteringP2P': 'Represent the title and abstract of a paper to identify its category',
            'MedrxivClusteringS2S': 'Represent the title of a paper to identify its category',
            'RedditClustering': 'Represent the titles to identify the topic or theme of Reddit posts',
            'RedditClusteringP2P': 'Represent the titles and posts to identify the topic or theme of Reddit posts',
            'StackExchangeClustering': 'Represent the titles to identify the topic or theme of StackExchange posts',
            'StackExchangeClusteringP2P': 'Represent the given paragraphs to identify the topic or theme of StackExchange posts',
            'TwentyNewsgroupsClustering': 'Represent news articles to identify their topic or theme.',
        },
        "PairClassification": {
            'SprintDuplicateQuestions': 'Retrieve duplicate questions from Sprint forum',
            'TwitterSemEval2015': 'Represent the tweet to find another similar tweet',
            'TwitterURLCorpus': 'Represent the tweet to find another similar tweet',
        },
        'Reranking': {
            # Questions from AskUbuntu with manual annotations marking pairs of questions as similar or dissimilar.
            'AskUbuntuDupQuestions': {
                'query': 'Represent the query to find a sematically similar query on the AskUbuntu community forum',
                'corpus': 'Represent the query to find a sematically similar query on the AskUbuntu community forum',
            },
            # Stack Overflow Duplicate Questions Task for questions with the tags Java, JavaScript and Python, ranking questions as duplicates or not
            'StackOverflowDupQuestions': {
                'query': 'Represent the query to find a sematically similar query on the StackOverflow community forums',
                'corpus': 'Represent the query to find a sematically similar query on the StackOverflow community forums',
            },
            # Both are titles, e.g. "Beauty eMakeup: A Deep Makeup Transfer System" matches with "Makeup like a superstar: Deep Localized Makeup Transfer Network"
            'SciDocsRR': {
                'query': 'Represent the title to find a similar scientific paper title',
                'corpus': 'Represent the title to find a similar scientific paper title',
            },
            # E.g. "Taylor Swift Says Scooter Braun, Scott Borchetta Are Blocking Her From Playing Old Hits at AMAs" matches with "Author Jennine Capó Crucet responds after white college students burn her book" but not with "How to Make Buttermilk At Home With Just 2 Ingredients"
            'MindSmallReranking': {
                'query': 'Represent the news query for retrieving relevant articles.',
                'corpus': 'Represent the news article.',
            },
        },
        'Retrieval': {
            ### Bio-Medical Information Retrieval ###
            # NFCorpus [7] contains natural language queries harvested from NutritionFacts (NF). We use the original splits provided alongside all content sources from NF (videos, blogs, and Q&A posts) as queries Q and annotated medical documents from PubMed as corpus T.
            'NFCorpus': [{
                'query': 'Represent the query to find a medical document relevant with it',
                'corpus': 'Represent this text of a medical document to find a query that it answers',
            },
            "From the question, provice a professional medical answer for it"
            ],
            # TREC-COVID [65] is an ad-hoc search challenge based on the CORD-19 dataset containing scientific articles related to the COVID-19 pandemic [69]. We include the July 16, 2020 version of CORD-19 dataset as corpus T and use the final cumulative judgements with query descriptions from the original task as queries Q.
            'TRECCOVID': [{
                'query': 'Represent the query on COVID-19 to retrieve documents that answer the query',
                'corpus': 'Represent the scientific article about COVID-19'
            },
            "Given a question about COVID-19, find a scientific article that answers the question"
            ],
            ### Open-domain Question Answering (QA) ###
            'MSMARCO': [{
                'query': 'Represent the web search query to find a passage that addresses it',
                'corpus': 'Represent the passage for finding a search query that it addresses',
            },
            "Generate a passage such that it can be matched with the following user's search query that it adequately answers."
            ],
            # Natural Questions [34] contains Google search queries and documents with paragraphs and answer spans within Wikipedia articles. We did not use the NQ version from ReQA [1] as it focused on queries having a short answer. As a result, we parsed the HTML of the original NQ dataset and include more complex development queries that often require a longer passage as answer compared to ReQA. We filtered out queries without an answer, or having a table as an answer, or with conflicting Wikipedia pages. We retain 2,681,468 passages as our corpus T and 3452 test queries Q from the original dataset.
            'NQ': [{
                'query': 'Represent the query to find an answer passage from a Wikipedia article that addresses it',
                'corpus': 'Represent the Wikipedia article passage to find a query that would be addressed by it',
            },
            "From the question, find a cutout from Wikipedia with the answer."
            ],
            # HotpotQA [76] contains multi-hop like questions which require reasoning over multiple paragraphs to find the correct answer. We include the original full-wiki task setting: utilizing processed Wikipedia passages as corpus T. We held out randomly sampled 5447 queries from training as our dev split. We use the original (paper) task’s development split as our test split Q.
            'HotpotQA': [{
                # Wikipedia Question
                'query': 'Represent the question to find documents that can help answer the question',
                # Wikipedia Articles
                'corpus': 'Represent the document to find a question that it relates to',
            },
            "Find a document that answers the following question from Wikipedia"
            ],
            # FiQA-2018 [44] Task 2 consists of opinion-based question-answering. We include financial data by crawling StackExchange posts under the Investment topic from 2009-2017 as our corpus T. We randomly sample out 500 and 648 queries Q from the original training split as dev and test splits.            
            'FiQA2018': [{
                'query': 'Represent the financial question to find user replies that best answer it',
                'corpus': 'Represent the financial post to find a question that it answers',
            },
            "Given this question in financial topic, get an answer for it."
            ],
            ### Argument Retrieval ###
            # ArguAna Counterargs Corpus [67] involves the task of retrieval of the best counterargument to an argument. We include pairs of arguments and counterarguments scraped from the online debate portal as corpus T. We consider the arguments present in the original test split as our queries Q.            
            'ArguAna': [{
                'query': 'Represent a claim, find passages that refute the claim',
                'corpus': 'Represent the passage to find a claim that it refutes',
            },
            "Given a claim, generate a passage that refutes it."
            ],
            # Touché-2020 [6] Task 1 is a conversational argument retrieval task. We use the conclusion as title and premise for arguments present in args.me [66] as corpus T. We include the shared Touché-2020 task data as our test queries Q. The original relevance judgements (qrels) file also included negative judgements (-2) for non-arguments present within the corpus, but for simplicity we substitute them as zero.            
            'Touche2020': [{
                'query': 'Represent a question to retrieve detailed and persuasive arguments that answer the question',
                'corpus': 'Represent an argument to retrieve a question that it takes a stance about',
            },
            "Answer the following question."
            ],
            ### Duplicate Question Retrieval ###
            # CQADupStack [25] is a popular dataset for research in community question-answering (cQA). The corpus T comprises of queries from 12 different StackExchange subforums: Android, English,Gaming, Gis, Mathematica, Physics, Programmers, Stats, Tex, Unix, Webmasters and Wordpress. We utilize the original test split for our queries Q, and the task involves retrieving duplicate query (title + body) for an input query title. We evaluate each StackExchange subforum separately and report the overall mean scores for all tasks in BEIR.            
            # Example query: Android chroot ubuntu - is it possible to get ubuntu to recognise usb devices
            # Example doc: I want to send files to android tablet with a application from PC. - I can send files directly to tablet (2.3 android OS) PC see it as a external usb drive. - But i can't send files to tablet (4.2 android OS), because PC see it as a portable media player.(MTP) - How can i fix this problem ? - How can show my device as a external drive? my application that sent files written via Delphi.
            # Example doc title: How can show android tablet as a external storage to PC?
            'CQADupstackTexRetrieval': [{
                'query': 'Represent a question to find detailed question descriptions from Stackexchange that are duplicates to the given question',
                'corpus': 'Represent a question to find detailed question descriptions from Stackexchange that are duplicates to the given question',
            },
            "Given this question, find a similar question from Stackexchange"
            ],
            # Quora Duplicate Questions dataset identifies whether two questions are duplicates. Quora originally released containing 404,290 question pairs. We add transitive closures to the original dataset. Further, we split it into train, dev, and test sets with a ratio of about 85%, 5% and 10% of the original pairs. We remove all overlaps between the splits and ensure that a question in one split of the dataset does not appear in any other split to mitigate the transductive classification problem [27]. We achieve 522,931 unique queries as our corpus T and 5,000 dev and 10,000 test queries Q respectively
            'QuoraRetrieval': [{
                'query': 'Represent the question to find another similar question on Quora',
                'corpus': 'Represent the question to find another simtlar question on Quora',
            },
            "Find a similar quora-style question"
            ],
            ### Entity Retrieval ###
            # DBPedia-Entity-v2 [21] is an established entity retrieval dataset. It contains a set of heterogeneous entity-bearing queries Q containing named entities, IR style keywords, and natural language queries. The task involves retrieving entities from the English part of DBpedia corpus T from October 2015. We randomly sample out 67 queries from the test split as our dev set.
            'DBPedia': [{
                'query': 'Represent the query to find relevant entity descriptions from DBPedia',
                'corpus': 'Represent the entity descriptions from DBPedia',
            },
            "Given a query, find a relevant entity description"
            ],
            ### Citation Prediction ###
            # SCIDOCS [9] contains a corpus T of 30K held-out pool of scientific papers. We consider the direct-citations (1 out of 7 tasks mentioned in the original paper) as the best suited task for retrieval evaluation in BEIR. The task includes 1k papers as queries Q with 5 relevant papers and 25 (randomly selected) uncited papers for each query.
            'SCIDOCS': [{
                'query': 'Represent the scientific paper title to find paper abstracts that are similar the given paper',
                'corpus': 'Represent the abstract of this scientific paper to find the title of another scientific paper on PubMed that similar the given paper',
            },
            "With the title, generate the corresponding scientific abstract."
            ],
            ### Fact Checking ###
            # FEVER [60] The Fact Extraction and VERification dataset is collected to facilitate the automatic fact checking. We utilize the original paper splits as queries Q and retrieve evidences from the pre-processed Wikipedia Abstracts (June 2017 dump) as our corpus T.
            'FEVER': [{
                'query': 'Represent the claim about to find documents that support or refute the claim',
                # Wikipedia Articles
                'corpus': 'Represent the documents to find a claim that it supports or refute',
            },
            "Given a claim, find a document that supports or refutes it."
            ],
            'ClimateFEVER': [{
                # Climate-based Claim
                'query': 'Represent the claim about climate change to find documents that support or refute the claim',
                # Wikipedia Articles
                'corpus': 'Represent the documents to find a claim about climate change that it supports or refute',
            },
            "Given a claim about climate change, find a document that supports or refutes it."
            ],
            # SciFact [68] verifies scientific claims using evidence from the research literature containing scientific paper abstracts. We use the original publicly available dev split from the task containing 300 queries as our test queries Q, and include all documents from the original dataset as our corpus T.
            'SciFact': [{
                'query': 'Represent the scientific claim to find a documents to support it',
                'corpus': 'Represent the scientific documents to find a claim that it supports',
            },
            "Given the fact, generate its justifications"
            ],
        },
        'STS': {
            # Other prompt candidates:
            # 'Represent the sentence to find another single-sentence casual post about the same topic',
            'STS12': 'Represent the sentence to find another sentence with the same meaning',
            'STS13': 'Represent the sentence to find another sentence with the same meaning',
            # For the English subtask, we exposed the systems to a diversity of testing scenarios, by preparing additional OntoNotesWordNet sense mappings and news headlines, as well as introducing new genres, including image descriptions, DEFT discussion forums, DEFT newswire, and tweet-newswire headline mappings
            'STS14': 'Represent the sentence to find another sentence with the same meaning',
            'STS15': 'Represent the sentence to find another sentence with the same meaning',
            'STS16': 'Represent the sentence to find another sentence with the same meaning',
            'STS17': 'Represent the sentence to find another sentence with the same meaning',
            'STS22': 'Represent the sentence to find another sentence with the same meaning',
            'BIOSSES': 'Represent the text to find another biological statement with the same meaning',
            # Sentences Involving Compositional Knowledge (SICK) contains a large number of sentence pairs (10 0000) that are lexically, syntactically and semantically rich.
            'SICK-R': 'Represent the sentence to find another sentence with the same meaning',
            'STSBenchmark': 'Represent the sentence to find another sentence with the same meaning',
        },
        'Summarization': {
            'SummEval': 'Represent the summary to find another semantically similar summary of the same news article',
        },        
    },
}

QUICK_EVAL = [
    # Classification
    "Banking77Classification",
    "EmotionClassification",
    # Clustering
    "MedrxivClusteringS2S",
    # PairClassification
    "TwitterSemEval2015",
    # Reranking
    "AskUbuntuDupQuestions",
    # Retrieval
    "ArguAna",
    "NFCorpus",
    "SciFact",
    # STS
    "BIOSSES",
    "STS17",
    "STSBenchmark",
    # Summarization
    "SummEval",
]

DTYPE_TO_TORCH_DTYPE = {
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
    'float16': torch.float16,
}

def get_gpus_max_memory(max_memory):
    max_memory = {i: max_memory for i in range(torch.cuda.device_count())}
    return max_memory

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True, help="Model name or path")
    parser.add_argument('--reranker_model_name_or_path', type=str, default=None, help="Reranker model name or path")
    parser.add_argument('--is_old', default=False, action='store_true', help="Use old model")
    parser.add_argument('--pretrained_type', type=str, required=True, help="Mistral/Meta-Llama/phi-1_5")
    parser.add_argument('--attn_implementation', default='sdpa', type=str, help="eager/sdpa/flash_attention_2")
    parser.add_argument('--use_bidirectional', action='store_true', help="Use bidirectional attention")
    parser.add_argument('--task_types', default=None, help="Comma separated. Default is None i.e. running all tasks")
    parser.add_argument('--task_names', default=None, help="Comma separated. Default is None i.e. running all tasks")
    parser.add_argument('--instruction_set', default="genclm", type=str, help="Instructions to use")
    parser.add_argument('--instruction_format', default="genclm", type=str, help="Formatting to use")
    parser.add_argument('--no_instruction', action='store_true', help="Do not use instructions")
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_length', default=None, type=int)
    parser.add_argument('--num_shots', default=None, type=int)
    parser.add_argument('--dtype', default='bfloat16', type=str)
    parser.add_argument('--output_folder', default=None, type=str)
    parser.add_argument('--overwrite_results', action='store_true')
    parser.add_argument('--pipeline_parallel', action='store_true')
    parser.add_argument('--rerank', action='store_true')
    parser.add_argument('--pooling_method', default='mean', type=str)
    parser.add_argument('--top_k', default=10, type=int)    
    return parser.parse_args()


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    args = get_args()

    model_name = args.model_name_or_path.rstrip('/').split('/')[-1]
    output_folder = args.output_folder if args.output_folder else f"results/{model_name}"
    if not args.overwrite_results:
        if (args.task_names is not None) and (len(args.task_names.split(",")) == 1) and os.path.exists(f"{output_folder}/{args.task_names.split(',')[0]}.json"):
            print(f"Skipping {args.task_names.split(',')[0]}")
            exit()
    
    model_kwargs = {
        "model_weights_name_or_path": args.model_name_or_path,
        "is_old": args.is_old,
        "pretrained_type": args.pretrained_type,
        "use_bidirectional": args.use_bidirectional,
        "normalized": False,
        "pooling_method": args.pooling_method,
        "torch_dtype": DTYPE_TO_TORCH_DTYPE.get(args.dtype, torch.bfloat16),
        "is_inference": True,
        "attn_implementation": args.attn_implementation,
    }
    if args.pipeline_parallel:
        model_kwargs["device_map"] = "auto"
        # model_kwargs["max_memory"] = get_gpus_max_memory("50GB")
        model_kwargs["offload_folder"] = "offload"

    kwargs = {"task_langs": ['en']}
    if args.task_names:
        kwargs["tasks"] = args.task_names.split(",")
    elif args.task_types:
        kwargs["task_types"] = args.task_types.split(",")
    tasks = [(t.metadata.name, t.metadata.type) for t in MTEB(**kwargs).tasks]

    for (task_name, task_type) in tasks:
        if task_name in ['MSMARCOv2', 'BigPatentClustering']:
            print('Skipping task: ' + task_name)
            continue
        
        if not args.no_instruction:
            if task_name.startswith("CQADupstack") and \
                "CQADupstackRetrieval" in SET_TO_TASK_TO_DS_TO_PROMPT[args.instruction_set][task_type]:
                instruction = SET_TO_TASK_TO_DS_TO_PROMPT[args.instruction_set][task_type]["CQADupstackRetrieval"]
            else:
                if task_name not in SET_TO_TASK_TO_DS_TO_PROMPT[args.instruction_set][task_type]:
                    print('Skipping task: ' + task_name)
                    continue
                instruction = SET_TO_TASK_TO_DS_TO_PROMPT[args.instruction_set][task_type][task_name]
            
            gen_prompt = None
            if isinstance(instruction, list):
                gen_prompt = "Craft a paragraph that addresses the query"
                instruction = instruction[0]
            
            if isinstance(instruction, dict):
                instruction = {k: v.strip(": \n") for k, v in instruction.items()}
            else:
                instruction = instruction.strip(": \n")
            
            print(f"{model_name} instruction for {task_name}: ", instruction)
            if gen_prompt:
                print(f"{model_name} gen_prompt for {task_name}: ", gen_prompt)

            model = GenCLMRetrieval(**model_kwargs)
            if args.max_length is not None:
                model.encode = partial(model.encode, max_length=args.max_length)
            if isinstance(instruction, dict):
                model.encode_queries = partial(model.encode_queries, instruction=instruction['query'])
                model.encode_corpus = partial(model.encode_corpus, instruction=instruction['corpus'])
            else:
                model.encode = partial(model.encode, instruction=instruction)
            
        eval_splits = ["test" if task_name not in ['MSMARCO'] else 'dev']
        evaluation = MTEB(tasks=[task_name], task_langs=['en'])
        save_predictions = True if task_name in SET_TO_TASK_TO_DS_TO_PROMPT[args.instruction_set]['Retrieval'] else False
        # if exit prediction_path and rerank is True, then rerank the predictions using the reranker model
        prediction_path = os.path.join(output_folder, f"{task_name}_predictions.json")
        if os.path.exists(prediction_path) and args.rerank:
            print("Skipping predictions for", task_name)
        else:
            evaluation.run(
                model,
                output_folder=output_folder,
                eval_splits=eval_splits,
                batch_size=args.batch_size,
                save_predictions=save_predictions,
                overwrite_results=args.overwrite_results,
            )
        if gen_prompt and args.rerank:
            # Clear the model and gpu memory to avoid OOM errors when loading a new model
            torch.cuda.empty_cache()
            del model
            if args.reranker_model_name_or_path is not None:
                model_kwargs["model_weights_name_or_path"] = args.reranker_model_name_or_path
            model = GenCLMReranker(**model_kwargs)
            if args.max_length is not None:
                model.predict = partial(model.predict, max_length=args.max_length)
            model.predict = partial(model.predict, instruction=gen_prompt)
            evaluation.run(
                model,
                output_folder=os.path.join(output_folder, "rerank"),
                eval_splits=eval_splits,
                batch_size=4,
                top_k=args.top_k,
                save_qrels=False,
                overwrite_results=args.overwrite_results,
                previous_results=prediction_path,
            )




                
