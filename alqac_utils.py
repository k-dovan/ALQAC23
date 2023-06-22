import logging
import json
from tqdm import tqdm
from typing import List

from haystack.document_stores import InMemoryDocumentStore
from haystack import Pipeline

def init_logger(logger_name: str, level: int):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)
    logger.addHandler(ch)

    return logger

# init logger
logger = init_logger('utils', logging.WARNING)

# ============================= data utils ====================================
# read law data from a json file to dict (with required format)
# {"content": "...", "meta": {"law_id": "05/2022/QH15", "article_id": "95"}}


def read_corpus(file_path: str) -> List[dict]:
    res = []
    with open(file_path, "r", encoding='utf-8') as f:
        data = json.load(f)
        if not isinstance(data, list):
            return res
        for law in tqdm(data):
            if not (law["id"] and law["articles"]):
                continue
            if not isinstance(law["articles"], list):
                continue
            for art in law["articles"]:
                if not (art["id"] and art["text"]):
                    continue
                item = {"content": art["text"], "meta": {
                    "law_id": law["id"], "article_id": art["id"]}}
                res.append(item)

    return res


def prepare_in_memory_dataset(file_paths: List[str]) -> InMemoryDocumentStore:
    document_store = InMemoryDocumentStore(
        use_bm25=True,
        similarity='dot_product',
        embedding_dim=768,
        use_gpu=True,
        progress_bar=True
    )
    # load data from json files
    data = []
    for file_path in file_paths:
        data.extend(read_corpus(file_path))

    # skip duplicate documents if exist
    document_store.write_documents(
        data, batch_size=1000, duplicate_documents="skip")

    return document_store


def read_json_sets(file_paths: List[str]) -> List[dict]:
    res = []
    for file_path in tqdm(file_paths, desc=f"Reading {file_paths}"):
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                continue
            res.extend(data)
    return res

# ================================ metrics =========================================
class ArticleIDs():
    def __init__(self, law_id, article_id):
        self.law_id, self.article_id = law_id, article_id

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
            getattr(other, 'law_id', None) == self.law_id and
            getattr(other, 'article_id', None) == self.article_id)
    
    def __hash__(self):
        return hash(self.law_id + str(self.article_id))

    def __str__(self) -> str:
        return f"{{law_id: {self.law_id}, article_id: {self.article_id}}}"


def evaluate_pipeline(eval_sets, pipeline: Pipeline, 
                      retrival_method: str, 
                      retriever_top_k: int = 100, 
                      ranker_top_k: int = 1,
                      evaluation_type: str = 'f2' # `f2`, `coverage`, `public_test`
                      ):
    """
    Evaluate the pipeline using `F2-metric` provided by ALQAC2023 on the `eval_sets` OR Evaluate how good the pipeline's retrieved documents cover ground-truth relevant articles.
    
    F2-metric formular:
        Precision(i) = the number of correctly retrieved articles of question i_th/the number of retrieved articles of question i_th
        
        Recall(i) = the number of correctly retrieved articles of question i_th/the number of relevant articles of question i_th
        
        F2(i) = ( 5 x Precision(i) x Recall(i) ) / ( 4Precision(i) + Recall(i) ) 
        
        F2 = average of (F2(i)

    """
    Precisions, Recalls, F2s = [], [], []
    coverages = []
    iter = 0
    for question in tqdm(eval_sets, desc="Reading questions in training sets"):
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
        
        logger.debug(f'relevant articles: {[str(art) for art in relevant_articles]}')

        if "Ranker" in pipeline._get_all_component_names():
            prediction = pipeline.run(
                query=question["text"],
                params={retrival_method: {"top_k": retriever_top_k}, "Ranker": {"top_k": ranker_top_k}}
            )
        else:
            prediction = pipeline.run(
                query=question["text"],
                params={retrival_method: {"top_k": retriever_top_k}}
            )

        retrieved_docs = prediction["documents"]
        if not len(retrieved_docs):
            continue
        
        iter +=1
        # build set of retrieved relevant articles
        retrieved_articles = set()
        for doc in retrieved_docs:
            if not doc.meta:
                continue
            if not (doc.meta["law_id"] and doc.meta["article_id"]):
                continue
            retrieved_articles.add(ArticleIDs(doc.meta["law_id"], doc.meta["article_id"]))
        
        logger.debug(f'retrieved articles: {[str(art) for art in retrieved_articles]}')

        if evaluation_type == 'f2':
            Precision_i = len(relevant_articles.intersection(retrieved_articles))/len(retrieved_articles)
            Recall_i = len(relevant_articles.intersection(retrieved_articles))/len(relevant_articles)
            
            logger.warn(f'iter: {iter}, precision_i: {Precision_i}')
            logger.warn(f'iter: {iter}, Recall_i {Recall_i}')

            if Precision_i == 0 or Recall_i == 0:
                F2_i = 0
            else:
                # calculate F2_i
                F2_i = ( 5 * Precision_i * Recall_i ) / ( 4 * Precision_i + Recall_i )        
            
            logger.warn(f'iter: {iter}, F2_i: {F2_i}')

            Precisions.append(Precision_i)
            Recalls.append(Recall_i)
            F2s.append(F2_i)
        elif evaluation_type == 'coverage':
            coverage_i = len(relevant_articles.intersection(retrieved_articles))/len(relevant_articles)
        
            logger.warn(f'iter: {iter}, coverage_i: {coverage_i}')

            coverages.append(coverage_i)
    
    if evaluation_type == 'f2':
        Precision = sum(Precisions)/len(Precisions)
        Recall = sum(Recalls)/len(Recalls)
        F2 = sum(F2s)/len(F2s)

        return Precision, Recall, F2
    
    elif evaluation_type == 'coverage':
        coverage = sum(coverages)/len(coverages)

        return coverage

def predict_public_test(public_test_set, pipeline: Pipeline, retrival_method: str, retriever_top_k: int = 100, ranker_top_k: int = 1):
    """
    Print/write to file public test result for submission.
    """
    iter = 0
    results = []

    for question in tqdm(public_test_set, desc="Reading public test questions"):
        if not (question["question_id"] and question["text"]):
            continue        

        if "Ranker" in pipeline._get_all_component_names():
            prediction = pipeline.run(
                query=question["text"],
                params={retrival_method: {"top_k": retriever_top_k}, "Ranker": {"top_k": ranker_top_k}}
            )
        else:
            prediction = pipeline.run(
                query=question["text"],
                params={retrival_method: {"top_k": retriever_top_k}}
            )

        retrieved_docs = prediction["documents"]
        if not len(retrieved_docs):
            continue
        
        iter +=1
        # build set of retrieved relevant articles
        retrieved_articles = []
        for doc in retrieved_docs:
            if not doc.meta:
                continue
            if not (doc.meta["law_id"] and doc.meta["article_id"]):
                continue
            
            retrieved_articles.append({"law_id": doc.meta["law_id"], "article_id": doc.meta["article_id"]})
        
        logger.debug(f'retrieved articles: {[str(art) for art in retrieved_articles]}')

        if len(retrieved_articles):
            result = {"question_id": question["question_id"], "relevant_articles": retrieved_articles}
            results.append(result)
            logger.warn(f"iter={iter},{result}")
    
    with open("public_test_prediction.json", 'w', encoding='utf-8') as f:
        json_object = json.dumps(results, indent=4, ensure_ascii=False)
        f.write(json_object)