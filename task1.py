
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

from alqac_utils import prepare_in_memory_dataset, read_eval_sets

logging.basicConfig(
    format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

DATASET_DIR = "../ALQAC_2023_training_data"

RETRIEVAL_METHODS = Enum('RETRIEVAL_METHODS',
                         ['TfidfRetriever', 'BM25Retriever',
                             'EmbeddingRetriever', 'DensePassageRetriever']
                         )

class ArticleIDs():
    def __init__(self, law_id, article_id):
        self.law_id, self.article_id = law_id, article_id

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
            getattr(other, 'law_id', None) == self.law_id and
            getattr(other, 'article_id', None) == self.article_id)
    
    def __hash__(self):
        return hash(self.law_id + str(self.article_id))
    
def build_retriever(document_store, retrieval_method: int):
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


def build_retriever_pipe(retriever, retrival_method: Enum, ranker_model_name: str = None) -> Pipeline:
    retriever_pipe = Pipeline()

    retriever_pipe.add_node(component=retriever,
                            name=retrival_method.name, inputs=["Query"])
    if not ranker_model_name:
        ranker = SentenceTransformersRanker(
            model_name_or_path=ranker_model_name)
        retriever_pipe.add_node(
            component=ranker, name="Ranker", inputs=[retrival_method.name])

    return retriever_pipe

def evaluate_pipeline(eval_sets, pipeline: Pipeline, retrival_method: Enum, retriever_top_k: int = 100, ranker_top_k: int = 1):
    """
    Evaluate the pipeline using `F2-metric` provided by ALQAC2023 on the `eval_sets`.
    
    F2-metric formular:
        Precision(i) = the number of correctly retrieved articles of question i_th/the number of retrieved articles of question i_th
        
        Recall(i) = the number of correctly retrieved articles of question i_th/the number of relevant articles of question i_th
        
        F2(i) = ( 5 x Precision(i) x Recall(i) ) / ( 4Precision(i) + Recall(i) ) 
        
        F2 = average of (F2(i)

    """
    Precisions, Recalls, F2s = [], [], []
    for question in eval_sets:
        if not (question["text"] and question["relevant_articles"]):
            continue
        if not isinstance(question["relevant_articles"], list):
            continue
        if not len(question["relevant_articles"]):
            continue
        
        # build set of ground-truth relevant articles
        relevant_articles = set()
        for art in question["relevant_articles"]:
            if not (art["law_id"] and art["article_id"]):
                continue
            relevant_articles.add(ArticleIDs(art["law_id"], art["article_id"]))

        prediction = pipeline.run(
            query=question["text"],
            params={retrival_method.name: {"top_k": retriever_top_k}, "Ranker": {"top_k": ranker_top_k}}
        )

        retrieved_docs = prediction["documents"]
        if not len(retrieved_docs):
            continue
        
        # build set of retrieved relevant articles
        retrieved_articles = set()
        for doc in retrieved_docs:
            if not doc["meta"]:
                continue
            if not (doc["meta"]["law_id"] and doc["meta"]["article_id"]):
                continue
            retrieved_articles.add(ArticleIDs(doc["meta"]["law_id"], doc["meta"]["article_id"]))

        Precision_i = len(relevant_articles.intersection(retrieved_articles))/len(retrieved_articles)
        Recall_i = len(relevant_articles.intersection(retrieved_articles))/len(relevant_articles)

        # calculate F2_i
        F2_i = ( 5 * Precision_i * Recall_i ) / ( 4 * Precision_i + Recall_i ) 

        Precisions.append(Precision_i)
        Recalls.append(Recall_i)
        F2s.append(F2_i)
    
    # Mean Average Precision/Recall/F2
    Precision = sum(Precisions)/len(Precisions)
    Recall = sum(Recalls)/len(Recalls)
    F2 = sum(F2s)/len(F2s)

    return Precision, Recall, F2

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

    # evaluate pipelines
    retrieval_method = RETRIEVAL_METHODS.BM25Retriever
    
    # build retriever
    retriever = build_retriever(document_store=document_store, retrieval_method=retrieval_method)

    # build retriver pipeline without Ranker
    pipeline = build_retriever_pipe(retriever=retriever, retrival_method=retrieval_method)



    # =========================================================================================================== 
    # # 2. retriever using BM25 algorithm alone
    # retriever = BM25Retriever(document_store=document_store)

    # # retrieve all relevant documents provided given a question
    # relevants = retriever.retrieve(
    #     query='Điều nào dưới đây nằm trong luật áp dụng giải quyết tranh chấp được quy định trong Luật Trọng tài thương mại?',
    #     top_k=1
    # )

    # print(relevants)

    # # 3. The Retriever-Ranker Pipeline
    # retriever_pipe = Pipeline()

    # ranker = SentenceTransformersRanker(
    #     model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2")

    # retriever_pipe.add_node(component=retriever,
    #                         name="BM25Retriever", inputs=["Query"])
    # retriever_pipe.add_node(
    #     component=ranker, name="Ranker", inputs=["BM25Retriever"])

    # prediction = retriever_pipe.run(
    #     query='Điều nào dưới đây nằm trong luật áp dụng giải quyết tranh chấp được quy định trong Luật Trọng tài thương mại?',
    #     params={"BM25Retriever": {"top_k": 100}, "Ranker": {"top_k": 1}}
    # )

    # print(prediction)

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
