
import os
import logging
import argparse
import pickle
from pprint import pprint
from typing import List
from enum import Enum

from haystack import Pipeline
from haystack.nodes import BM25Retriever, TfidfRetriever, EmbeddingRetriever, DensePassageRetriever, \
    SentenceTransformersRanker, FARMReader
from haystack.utils import print_answers
from haystack.document_stores import InMemoryDocumentStore
from haystack.schema import Document

from alqac_utils import *
from task1 import *

logger = init_logger('dig_semantic_pairs', logging.ERROR)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--ranker_model', type=str, default="saved_models/mmarco-mMiniLMv2-L12-H384-v1-VN-LegalQA-bm25-512-10", help='cross-encoder model to rank documents.')
    parser.add_argument('-e', '--ranker_top_k', type=int, default=50, help='Number of returned documents to extract by Ranker.')

    return parser.parse_args()

if __name__ == "__main__":

    # parse arguments from commandline
    args = parse_arguments()

    ranker_model = args.ranker_model
    ranker_top_k = args.ranker_top_k

    ranker_model_name = os.path.basename(ranker_model)

    # read corpus_dict as reference
    corpus_data = json.load(open(f"{DATASET_DIR}/generated_data/corpus_dict.json"))

    # 1. prepare corpora and train sets 
    # load all corpora
    document_store = prepare_in_memory_dataset(file_paths=[path for path in CORPORA.values()])

    # load all train sets
    train_sets = read_json_sets([path for path in EVAL_SETS.values()])
    
    # build ranker pipeline    
    pipeline = retriever_pipe = Pipeline()
    retriever = BM25Retriever(document_store=document_store)
    pipeline.add_node(component=retriever,
                            name="BM25Retriever", inputs=["Query"])
    ranker = SentenceTransformersRanker(
            model_name_or_path=ranker_model,
            scale_score=False,       # use raw score
            use_gpu=True
            )
    pipeline.add_node(
            component=ranker, name="Ranker", inputs=["BM25Retriever"])

    # build finetuning dataset for cross-encoder model (used as ranker in our pipeline)
    query_text_pairs = []
    iter = 0
    for question in tqdm(train_sets, desc="Building query_doc pairs"):
        if not ("text" in question.keys() and "relevant_articles" in question.keys()):
            continue   
        if not isinstance(question["relevant_articles"], list):
            continue
        if not len(question["relevant_articles"]):
            continue
        
        relevant_keys = []
        # build set of [query-ground-truth relevant article] pairs
        for art in question["relevant_articles"]:
            if not ("law_id" in art.keys() and "article_id" in art.keys()):
                continue
            
            # get article text from corpus
            key = f'{art["law_id"]}_{art["article_id"]}'
            if key not in corpus_data.keys():
                continue
            relevant_keys.append(key)

            positive_pair = {}
            positive_pair["question"] = question["text"]
            positive_pair["document"] = corpus_data[key]
            positive_pair["relevant"] = 1
            query_text_pairs.append(positive_pair)

            logger.info(f'iter={iter}, question={positive_pair["question"]}, text={positive_pair["document"]}, relevant={positive_pair["relevant"]}')

        prediction = pipeline.run(
                query=question["text"],
                params={"BM25Retriever": {"top_k": 500}, "Ranker": {"top_k": args.ranker_top_k}}
        )            

        retrieved_docs = prediction["documents"]
        if not len(retrieved_docs):
            continue
        
        iter +=1
        for doc in retrieved_docs:
            if not doc.meta:
                continue
                        
            if not (doc.meta["law_id"] and doc.meta["article_id"]):
                continue
            key = f'{doc.meta["law_id"]}_{doc.meta["article_id"]}'
            if key in relevant_keys:
                continue

            negative_pair = {}
            negative_pair["question"] = question["text"]
            negative_pair["document"] = doc.content
            negative_pair["relevant"] = 0
            query_text_pairs.append(negative_pair)

            logger.info(f'iter={iter}, question={negative_pair["question"]}, text={negative_pair["document"]}, relevant={negative_pair["relevant"]}')
    
    save_path = f"{DATASET_DIR}/generated_data"
    os.makedirs(save_path, exist_ok=True)
    
    with open(os.path.join(save_path, f"qrel_pairs_{ranker_model_name}_top{ranker_top_k}.json"), 'w', encoding='utf-8') as f:
        json_object = json.dumps(query_text_pairs, indent=4, ensure_ascii=False)
        f.write(json_object) 