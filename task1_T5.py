
import logging
import argparse
from pprint import pprint
from typing import List
from enum import Enum

from haystack import Pipeline
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import BM25Retriever, TfidfRetriever, EmbeddingRetriever, DensePassageRetriever, \
    SentenceTransformersRanker, FARMReader
from haystack.utils import print_answers
from haystack.document_stores import InMemoryDocumentStore
from haystack.schema import Document

import pandas as pd
import pyterrier as pt
from pyterrier_t5 import MonoT5ReRanker
from alqac_utils import *

logger = init_logger('task1_T5', logging.INFO)

DATASET_DIR = "../ALQAC_2023_training_data"

CORPUS_CHOICES = ['ALQAC2023', 'ALQAC2022', 'Zalo']
CORPORA = {
        CORPUS_CHOICES[0]: f'{DATASET_DIR}/law.json',
        CORPUS_CHOICES[1]: f'{DATASET_DIR}/additional_data/ALQAC_2022_training_data/law.json',
        CORPUS_CHOICES[2]: f'{DATASET_DIR}/additional_data/zalo/zalo_corpus.json'
}
EVAL_SETS = {
    CORPUS_CHOICES[0]: f'{DATASET_DIR}/train.json',
    CORPUS_CHOICES[1]: f'{DATASET_DIR}/additional_data/ALQAC_2022_training_data/question.json',
    CORPUS_CHOICES[2]: f'{DATASET_DIR}/additional_data/zalo/zalo_question.json'
}

RETRIEVAL_CHOICES = ['TfidfRetriever', 'BM25Retriever']

RANKER_MODELS = [
    'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',       # 85% coverage
    ]
    
def build_retriever(document_store, retrieval_method):
    retriever = None

    if retrieval_method == 'TfidfRetriever':
        retriever = TfidfRetriever(document_store=document_store)
    elif retrieval_method == 'BM25Retriever':
        retriever = BM25Retriever(document_store=document_store)
    elif retrieval_method == 'EmbeddingRetriever':
        retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
            model_format="sentence_transformers",
            use_gpu=True
        )
        document_store.update_embeddings(retriever)
    elif retrieval_method == 'DensePassageRetriever':
        retriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
            passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
            use_gpu=True
            )
        document_store.update_embeddings(retriever)
    return retriever

def ranker_T5(ranking_dataframes: pd.DataFrame, ranker_top_k: int = 1):
    df_with_rank = monoT5.transform(ranking_dataframes)

    # return the top_k highest relevance docs
    return df_with_rank.sort_values('rank', ascending=True).head(ranker_top_k)


def build_retriever_pipe(retriever, retrival_method: str) -> Pipeline:
    retriever_pipe = Pipeline()

    retriever_pipe.add_node(component=retriever,
                            name=retrival_method, inputs=["Query"])

    return retriever_pipe

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus', type=str, help='Chosen corpus to work on.', choices=CORPUS_CHOICES, default=CORPUS_CHOICES[0])
    parser.add_argument('-m', '--retrieval_method', type=str, help='Retrieval method to use.', choices=RETRIEVAL_CHOICES, default=RETRIEVAL_CHOICES[0])
    parser.add_argument('-e', '--retriever_top_k', type=int, help='Number of retrieved documents to extract by Retriever.', default=50)
    parser.add_argument('-n', '--ranker_model', type=str, help='Model name or path for the Ranker', choices=RANKER_MODELS, default=RANKER_MODELS[0])
    parser.add_argument('-a', '--ranker_top_k', type=int, help='Number of retrieved documents to extract by Ranker.', default=3)  
    parser.add_argument('-i', '--print_metric', help='Print F2-metric result.', action='store_true')      
    parser.add_argument('-o', '--print_coverage', help='Print coverage result.', action='store_true')      
    parser.add_argument('-p', '--print_public_test', help='Print and write to file public test result.', action='store_true')    

    return parser.parse_args()

if __name__ == "__main__":
    
    pt.init()
    monoT5 = MonoT5ReRanker()

    # parse arguments from commandline
    args = parse_arguments()

    corpus = args.corpus
    retrieval_method = args.retrieval_method
    retriever_top_k = args.retriever_top_k
    ranker_model = args.ranker_model
    ranker_top_k = args.ranker_top_k
    print_metric = args.print_metric
    print_coverage = args.print_coverage
    print_public_test = args.print_public_test

    # 1. prepare corpus and eval sets 
    # load corpus
    corpus_path = CORPORA[corpus]
    document_store = prepare_in_memory_dataset(file_paths=[corpus_path])

    # load eval sets
    eval_path = EVAL_SETS[corpus]
    eval_sets = read_json_sets([eval_path])
    
    # build retriever
    retriever = build_retriever(document_store=document_store, retrieval_method=retrieval_method)
    
    retriever_pipe = build_retriever_pipe(retriever=retriever, 
                                    retrival_method=retrieval_method
                                    )

    if print_coverage:
        # evaluate pipeline with own_defined `coverage` metric
        coverage = evaluate_pipeline(eval_sets=eval_sets, 
                                    pipeline=retriever_pipe, 
                                    retrival_method=retrieval_method,
                                    own_ranker=ranker_T5,
                                    retriever_top_k=retriever_top_k,
                                    ranker_top_k=ranker_top_k,
                                    evaluation_type='coverage'
                            )

        logger.info(f"Retriever: {retrieval_method}")
        logger.info(f"Top {retriever_top_k} retrieved articles cover {100* coverage:.2f}% ground-truth relevant articles.")
    
    elif print_metric:
        # evaluate pipeline with provided F2-metric
        Precision, Recall, F2 = evaluate_pipeline(eval_sets=eval_sets, 
                                                    pipeline=retriever_pipe, 
                                                    retrival_method=retrieval_method,
                                                    own_ranker=ranker_T5,
                                                    retriever_top_k=retriever_top_k,
                                                    ranker_top_k=ranker_top_k,
                                                    evaluation_type='f2'
                                                    )

        logger.info(f"Precision: {Precision}, Recall: {Recall}, F2: {F2}")
    
    elif print_public_test:
        # read public test from json
        public_test_set = read_json_sets([f'{DATASET_DIR}/public_test.json'])

        # write the pridiction result to file for submission
        predict_public_test(public_test_set=public_test_set, 
                            pipeline=retriever_pipe, 
                            retrival_method=retrieval_method,
                            own_ranker=ranker_T5,
                            retriever_top_k=retriever_top_k,
                            ranker_top_k=ranker_top_k
                        )      
       