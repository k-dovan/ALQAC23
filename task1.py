
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

from alqac_utils import *

logger = init_logger('task1', logging.INFO)

DATASET_DIR = "ALQAC_2023_training_data"
SAVED_PATH = f'{DATASET_DIR}/cleaned_corpus/'

CORPUS_CHOICES = ['ALQAC2023', 'ALQAC2022', 'Zalo']
CORPORA = {
        CORPUS_CHOICES[0]: f'{DATASET_DIR}/law.json',
        CORPUS_CHOICES[1]: f'{DATASET_DIR}/additional_data/ALQAC_2022_training_data/law.json',
        CORPUS_CHOICES[2]: f'{DATASET_DIR}/additional_data/zalo/zalo_corpus.json'
}
RETRIEVER_CLEANED_CORPORA = {
        CORPUS_CHOICES[0]: f'{SAVED_PATH}/law_2023_cleaned.json',
        CORPUS_CHOICES[1]: f'{SAVED_PATH}/law_2022_cleaned.json',
        CORPUS_CHOICES[2]: f'{SAVED_PATH}/law_zalo_cleaned.json'
}

EVAL_SETS = {
    CORPUS_CHOICES[0]: f'{DATASET_DIR}/train.json',
    CORPUS_CHOICES[1]: f'{DATASET_DIR}/additional_data/ALQAC_2022_training_data/question.json',
    CORPUS_CHOICES[2]: f'{DATASET_DIR}/additional_data/zalo/zalo_question.json'
}

RETRIEVAL_CHOICES = ['BM25Retriever', 'TfidfRetriever']

RANKER_MODELS = [
    'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',       # 85% coverage
    'saved_models/mmarco-mMiniLMv2-L12-H384-v1-VN-LegalQA-bm25',    
    'saved_models/mmarco-mMiniLMv2-L12-H384-v1-VN-LegalQA-bm25-512-10'
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


def build_retriever_pipe(retriever, retrival_method: str, ranker_model_name: str = None) -> Pipeline:
    retriever_pipe = Pipeline()

    retriever_pipe.add_node(component=retriever,
                            name=retrival_method, inputs=["Query"])
    if ranker_model_name is not None:
        ranker = SentenceTransformersRanker(
            model_name_or_path=ranker_model_name,
            scale_score=False       # use raw score
            )
        retriever_pipe.add_node(
            component=ranker, name="Ranker", inputs=[retrival_method])

    return retriever_pipe

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus', type=str, help='Chosen corpus to work on.', choices=CORPUS_CHOICES, default=CORPUS_CHOICES[0])
    parser.add_argument('-m', '--retrieval_method', type=str, help='Retrieval method to use.', choices=RETRIEVAL_CHOICES, default=RETRIEVAL_CHOICES[0])
    parser.add_argument('-e', '--retriever_top_k_range', type=str, help='Range of number of retrieved documents to extract by Retriever.', default="15:100:5")
    parser.add_argument('-b', '--best_configs_top_k', type=int, help='Number of pipeline configuration to save.', default=5)
    parser.add_argument('-n', '--ranker_model', type=str, help='Model name or path for the Ranker', choices=RANKER_MODELS, default=RANKER_MODELS[0])
    parser.add_argument('-a', '--ranker_top_k', type=int, help='Number of retrieved documents to extract by Ranker.', default=3)
    parser.add_argument('-w', '--with_ranker', help='Use Ranker along with Retriever.', action='store_true')    
    parser.add_argument('-i', '--print_metric', help='Print F2-metric result.', action='store_true')      
    parser.add_argument('-o', '--print_coverage', help='Print coverage result.', action='store_true')      
    parser.add_argument('-p', '--print_public_test', help='Print and write to file public test result.', action='store_true')    

    return parser.parse_args()

if __name__ == "__main__":

    # parse arguments from commandline
    args = parse_arguments()

    corpus = args.corpus
    retrieval_method = args.retrieval_method
    retriever_top_k_range = args.retriever_top_k_range
    best_configs_top_k = args.best_configs_top_k
    ranker_model = args.ranker_model
    ranker_top_k = args.ranker_top_k
    with_ranker = args.with_ranker
    print_metric = args.print_metric
    print_coverage = args.print_coverage
    print_public_test = args.print_public_test

    # 1. prepare corpus and eval sets 
    # load corpus
    corpus_path = CORPORA[corpus]
    cleaned_corpus_path = RETRIEVER_CLEANED_CORPORA[corpus]
    # read from raw corpus
    # document_store = prepare_in_memory_dataset(file_paths=[corpus_path])
    # read from cleaned corpus with correct format for haystack
    document_store = prepare_in_memory_dataset_with_cleaned_corpus(file_paths=[cleaned_corpus_path])

    # load eval sets
    eval_path = EVAL_SETS[corpus]
    eval_sets = read_json_sets([eval_path])
    
    # build retriever
    retriever = build_retriever(document_store=document_store, retrieval_method=retrieval_method)


    ranker_model_name= ranker_model if with_ranker else None
    
    pipeline = build_retriever_pipe(retriever=retriever, 
                                    retrival_method=retrieval_method,
                                    ranker_model_name=ranker_model_name
                                    )
    
    # parse range of retriever_top_k values
    range_args = retriever_top_k_range.split(':')
    range_args = [int(arg) for arg in range_args]
    logger.info(f"range_args: {range_args}")
    coverages_with_configs = []
    for retriever_top_k in range(range_args[0],range_args[1], range_args[2]):
        logger.info(f"**** Retriever_top_k value: {retriever_top_k} ****")
        if print_coverage:
            # evaluate pipeline with own_defined `coverage` metric
            coverage = evaluate_pipeline(eval_sets=eval_sets, 
                                        pipeline=pipeline, 
                                        retrival_method=retrieval_method,
                                        retriever_top_k=retriever_top_k,
                                        ranker_top_k=ranker_top_k,
                                        evaluation_type='coverage'
                                )

            logger.info(f"Retriever: {retrieval_method}")
            logger.info(f"Top {retriever_top_k} retrieved articles cover {100* coverage:.2f}% ground-truth relevant articles.")
            result = {"coverage": coverage, "retrieval_method": retrieval_method, "retriever_top_k": retriever_top_k, "ranker_model_name": ranker_model_name, "ranker_top_k": ranker_top_k}
            coverages_with_configs.append(result)
        
        elif print_metric:
            # evaluate pipeline with provided F2-metric
            Precision, Recall, F2 = evaluate_pipeline(eval_sets=eval_sets, 
                                                        pipeline=pipeline, 
                                                        retrival_method=retrieval_method,
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
                                pipeline=pipeline, 
                                retrival_method=retrieval_method,
                                retriever_top_k=retriever_top_k,
                                ranker_top_k=ranker_top_k
                            ) 
    if print_coverage:
        # extract best_configs_top_k configs
        top_best_configs = []
        
        if len(coverages_with_configs) > best_configs_top_k:
            top_best_configs = sorted(coverages_with_configs, key = lambda x: x['coverage'], reverse=True)[:best_configs_top_k]
        else:
            top_best_configs = sorted(coverages_with_configs, key = lambda x: x['coverage'], reverse=True)

        with open(f"experiment_resources/best_configs_for_coverage_{retrieval_method}_[{range_args[0]}-{range_args[1]}-{range_args[2]}]_{ranker_top_k}.json", "w+", encoding="utf-8") as f:
            json_object = json.dumps(top_best_configs, indent=4, ensure_ascii=False)
            f.write(json_object)
       