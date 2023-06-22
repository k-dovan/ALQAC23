
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

logger = init_logger('prepare_finetuning_data', logging.WARN)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--retrieval_method', type=str, help='Retrieval method to use.', choices=RETRIEVAL_CHOICES, default=RETRIEVAL_CHOICES[0])
    parser.add_argument('-e', '--retriever_top_k', type=int, help='Number of retrieved documents to extract by Retriever.', default=50)

    return parser.parse_args()

if __name__ == "__main__":

    # parse arguments from commandline
    args = parse_arguments()

    retrieval_method = args.retrieval_method
    retriever_top_k = args.retriever_top_k

    # 1. prepare corpora and train sets 
    # load all corpora
    document_store = prepare_in_memory_dataset(file_paths=[path for path in CORPORA.values()])

    # load all train sets
    train_sets = read_json_sets([path for path in EVAL_SETS.values()])
    
    # build retriever
    retriever = build_retriever(document_store=document_store, retrieval_method=retrieval_method)
    
    pipeline = build_retriever_pipe(retriever=retriever, 
                                    retrival_method=retrieval_method
                                    )

    # build finetuning dataset for cross-encoder model (used as ranker in our pipeline)
    query_text_pairs = []
    iter = 0
    for question in tqdm(train_sets, desc="Building query_doc pairs"):
        if not (question["question_id"] and question["text"]):
            continue   
        if not isinstance(question["relevant_articles"], list):
            continue
        if not len(question["relevant_articles"]):
            continue
        
        # build set of [query-ground-truth relevant article] pairs
        for art in question["relevant_articles"]:
            if not (art["law_id"] and art["article_id"]):
                continue
            filters = {
                "$and": {
                    "law_id": {"$eq": art["law_id"]},
                    "article_id": {"$eq": art["article_id"]}
                }
            }
            docs = document_store.get_all_documents(filters=filters)
            if not len(docs):
                continue

            positive_pair = {}
            positive_pair["question"] = question["text"]
            positive_pair["document"] = docs[0].content
            positive_pair["relevant"] = 1
            query_text_pairs.append(positive_pair)

            logger.warn(f'iter={iter}, question={positive_pair["question"]}, text={positive_pair["document"]}, relevant={positive_pair["relevant"]}')

        prediction = pipeline.run(
                query=question["text"],
                params={retrieval_method: {"top_k": retriever_top_k}}
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
            negative_pair = {}
            negative_pair["question"] = question["text"]
            negative_pair["document"] = doc.content
            negative_pair["relevant"] = 0
            query_text_pairs.append(negative_pair)

            logger.warn(f'iter={iter}, question={negative_pair["question"]}, text={negative_pair["document"]}, relevant={negative_pair["relevant"]}')
    
    save_path = f"{DATASET_DIR}/generated_finetuning_data"
    os.makedirs(save_path, exist_ok=True)
    method = ""
    if retrieval_method == "BM25Retriever":
        method = "bm25"
    elif retrieval_method == "TfidfRetriever":
        method = "tfidf"
    
    with open(os.path.join(save_path, f"query_doc_pairs_{method}_top{retriever_top_k}.json"), 'w', encoding='utf-8') as f:
        json_object = json.dumps(query_text_pairs, indent=4, ensure_ascii=False)
        f.write(json_object) 