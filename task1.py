
import logging
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

from alqac_utils import *

logger = init_logger('task1', logging.INFO)

DATASET_DIR = "../ALQAC_2023_training_data"

RETRIEVAL_METHODS = Enum('RETRIEVAL_METHODS',
                         ['TfidfRetriever', 'BM25Retriever',
                          'EmbeddingRetriever', 'DensePassageRetriever'
                         ]
                         )
    
def build_retriever(document_store, retrieval_method):
    retriever = None

    if retrieval_method == RETRIEVAL_METHODS.TfidfRetriever:
        retriever = TfidfRetriever(document_store=document_store)
    elif retrieval_method == RETRIEVAL_METHODS.BM25Retriever:
        retriever = BM25Retriever(document_store=document_store)
    elif retrieval_method == RETRIEVAL_METHODS.EmbeddingRetriever:
        retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
            model_format="sentence_transformers"
        )
        document_store.update_embeddings(retriever)
    elif retrieval_method == RETRIEVAL_METHODS.DensePassageRetriever:
        retriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
            passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base")
        document_store.update_embeddings(retriever)
    return retriever


def build_retriever_pipe(retriever, retrival_method: str, ranker_model_name: str = None) -> Pipeline:
    retriever_pipe = Pipeline()

    retriever_pipe.add_node(component=retriever,
                            name=retrival_method, inputs=["Query"])
    if ranker_model_name is not None:
        ranker = SentenceTransformersRanker(
            model_name_or_path=ranker_model_name)
        retriever_pipe.add_node(
            component=ranker, name="Ranker", inputs=[retrival_method])

    return retriever_pipe

if __name__ == "__main__":

    # 1. prepare corpus and eval sets
    corpus_paths = [
        f'{DATASET_DIR}/law.json',
        # f'{DATASET_DIR}/additional_data/ALQAC_2022_training_data/law.json',
        # f'{DATASET_DIR}/additional_data/zalo/zalo_corpus.json'
    ]
    eval_paths = [
        f'{DATASET_DIR}/train.json',
        # f'{DATASET_DIR}/additional_data/ALQAC_2022_training_data/question.json',
        # f'{DATASET_DIR}/additional_data/zalo/zalo_question.json'
    ]

    # load corpus datasets
    document_store = prepare_in_memory_dataset(file_paths=corpus_paths)

    # load eval sets
    eval_sets = read_eval_sets(eval_paths)

    # evaluate pipelines
    retrieval_method = RETRIEVAL_METHODS.TfidfRetriever
    
    # build retriever
    retriever = build_retriever(document_store=document_store, retrieval_method=retrieval_method)

    # =================================== without ranker =========================================
    # =========================================================================================
    # build retriver pipeline without Ranker
    pipeline = build_retriever_pipe(retriever=retriever, 
                                    retrival_method=retrieval_method.name)

    retriever_top_k = 5
    # evaluate pipeline with own_defined `coverage` metric
    coverage = evaluate_pipeline(eval_sets=eval_sets, 
                                pipeline=pipeline, 
                                retrival_method=retrieval_method.name,
                                retriever_top_k=retriever_top_k
                        )

    logger.info(f"Retriever: {retrieval_method.name}")
    logger.info(f"Top {retriever_top_k} retrieved articles cover {100* coverage:.2f}% ground-truth relevant articles.")

    # # evaluate pipeline with provided F2-metric
    # Precision, Recall, F2 = f2_metric(eval_sets=eval_sets, 
    #                                           pipeline=pipeline, 
    #                                           retrival_method=retrieval_method.name,
    #                                           retriever_top_k=1
    #                                           )

    # logger.info(f"Precision: {Precision}, Recall: {Recall}, F2: {F2}")

    # =================================== with ranker =========================================
    # =========================================================================================
    # # build retriver pipeline with Ranker
    # pipeline = build_retriever_pipe(retriever=retriever, 
    #                                 retrival_method=retrieval_method.name,
    #                                 ranker_model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")
    
    # retriever_top_k = 100
    # ranker_top_k = 5
    # # evaluate pipeline with own_defined `coverage` metric
    # coverage = evaluate_pipeline(eval_sets=eval_sets, 
    #                             pipeline=pipeline, 
    #                             retrival_method=retrieval_method.name,
    #                             retriever_top_k=retriever_top_k,
    #                             ranker_top_k=ranker_top_k
    #                             )

    # logger.info(f"Top {retriever_top_k} Retriever, top {ranker_top_k} Ranker retrieved articles cover {100* coverage:.2f}% ground-truth relevant articles.")

    # # evaluate pipeline with provided F2-metric
    # Precision, Recall, F2 = f2_metric(eval_sets=eval_sets, 
    #                                           pipeline=pipeline, 
    #                                           retrival_method=retrieval_method.name,
    #                                           retriever_top_k=1
    #                                           )

    # logger.info(f"Precision: {Precision}, Recall: {Recall}, F2: {F2}")