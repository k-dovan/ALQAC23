
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

from ALQAC23.alqac_utils import prepare_in_memory_dataset, read_eval_sets

logging.basicConfig(
    format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

DATASET_DIR = "../ALQAC_2023_training_data"

RETRIEVAL_METHODS = Enum('RETRIEVAL_METHODS',
                         ['TfidfRetriever', 'BM25Retriever',
                             'EmbeddingRetriever', 'DensePassageRetriever']
                         )


def build_retriever(document_store, retrieval_method: str):
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
    elif retrieval_method == RETRIEVAL_METHODS.DensePassageRetriever:
        retriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
            passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base")

    return retriever


def build_retriever_pipe(retriever, retrival_method, ranker_model_name: str = None) -> Pipeline:
    retriever_pipe = Pipeline()

    retriever_pipe.add_node(component=retriever,
                            name=retrival_method, inputs=["Query"])
    if not ranker_model_name:
        ranker = SentenceTransformersRanker(
            model_name_or_path=ranker_model_name)
        retriever_pipe.add_node(
            component=ranker, name="Ranker", inputs=[retrival_method])

    return retriever_pipe


if __name__ == "__main__":

    # 1. prepare corpus and eval sets
    corpus_paths = [f'{DATASET_DIR}/law.json',
                    #   f'{DATASET_DIR}/additional_data/ALQAC_2022_training_data/law.json',
                    #   f'{DATASET_DIR}/additional_data/zalo/zalo_corpus.json'
                    ]
    eval_paths = [f'{DATASET_DIR}/train.json',
                  #   f'{DATASET_DIR}/additional_data/ALQAC_2022_training_data/question.json',
                  #   f'{DATASET_DIR}/additional_data/zalo/zalo_question.json'
                  ]

    # load corpus datasets
    document_store = prepare_in_memory_dataset(file_paths=corpus_paths)

    # load eval sets
    eval_sets = read_eval_sets(eval_paths)

    # 2. retriever using BM25 algorithm alone
    retriever = BM25Retriever(document_store=document_store)

    # retrieve all relevant documents provided given a question
    relevants = retriever.retrieve(
        query='Điều nào dưới đây nằm trong luật áp dụng giải quyết tranh chấp được quy định trong Luật Trọng tài thương mại?',
        top_k=1
    )

    print(relevants)

    # 3. The Retriever-Ranker Pipeline
    retriever_pipe = Pipeline()

    ranker = SentenceTransformersRanker(
        model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2")

    retriever_pipe.add_node(component=retriever,
                            name="BM25Retriever", inputs=["Query"])
    retriever_pipe.add_node(
        component=ranker, name="Ranker", inputs=["BM25Retriever"])

    prediction = retriever_pipe.run(
        query='Điều nào dưới đây nằm trong luật áp dụng giải quyết tranh chấp được quy định trong Luật Trọng tài thương mại?',
        params={"BM25Retriever": {"top_k": 100}, "Ranker": {"top_k": 1}}
    )

    print(prediction)

    # ============================================================================================
    # # 4. The Retriever-Reader Pipeline
    # reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

    # qa_pipe = ExtractiveQAPipeline(reader, retriever)
    # # Asking a question
    # prediction = qa_pipe.run(
    #     query="Who is the father of Arya Stark?",
    #     params={
    #         "Retriever": {"top_k": 1},
    #         "Reader": {"top_k": 1}
    #     }
    # )

    # # Print out the answers the pipeline returned
    # pprint(prediction)

    # # Simplify the printed answers
    # print_answers(
    #     prediction,
    #     details="minimum" ## Choose from `minimum`, `medium`, and `all`
    # )
