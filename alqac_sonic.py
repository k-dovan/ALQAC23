import json
import logging
from pprint import pprint
from typing import List

from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, SentenceTransformersRanker, FARMReader
from haystack import Pipeline
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

DATASET_DIR = "ALQAC_2023_training_data"

# read law data from a json file to dict (with required format)
# {"content": "...", "meta": {"law_id": "05/2022/QH15", "article_id": "95"}}
def read_data(file_path: str) -> dict:
  res = []
  with open(file_path, "r", encoding='utf-8') as f:
    data = json.load(f)
    if not isinstance(data, list):
      return res
    for law in data:
      if not (law["id"] and law["articles"]):
        continue
      if not isinstance(law["articles"], list):
        continue
      for art in law["articles"]:
        if not (art["id"] and art["text"]):
          continue
        item = {"content": art["text"], "meta": {"law_id": law["id"], "article_id": art["id"]}}
        res.append(item)

  return res

def prepare_in_memory_dataset(file_paths: List[str]) -> InMemoryDocumentStore:
    document_store = InMemoryDocumentStore(use_bm25=True)
    # load data from json files
    data = []
    for file_path in file_paths:      
        data.extend(read_data(file_path))
    
    # skip duplicate documents if exist
    document_store.write_documents(data, batch_size=1000, duplicate_documents="skip")

    return document_store

if __name__ == "__main__":

    # 1. prepare dataset
    file_paths = [f'{DATASET_DIR}/law.json',
                #   f'{DATASET_DIR}/additional_data/ALQAC_2022_training_data/law.json',
                #   f'{DATASET_DIR}/additional_data/zalo/zalo_corpus.json'
                  ]
    document_store = prepare_in_memory_dataset(file_paths=file_paths)

    # 2. retriever using BM25 algorithm alone
    retriever = BM25Retriever(document_store=document_store)

    # retrieve all relevant documents provided given a question
    relevants = retriever.retrieve(
        query = 'Điều nào dưới đây nằm trong luật áp dụng giải quyết tranh chấp được quy định trong Luật Trọng tài thương mại?',
        top_k = 1
    )

    print (relevants)

    # 3. The Retriever-Ranker Pipeline
    retriever_pipe = Pipeline()

    ranker = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2")

    retriever_pipe.add_node(component=retriever, name="BM25Retriever", inputs=["Query"])
    retriever_pipe.add_node(component=ranker, name="Ranker", inputs=["BM25Retriever"])

    prediction = retriever_pipe.run(
        query='Điều nào dưới đây nằm trong luật áp dụng giải quyết tranh chấp được quy định trong Luật Trọng tài thương mại?',
        params={"BM25Retriever": {"top_k": 100}, "Ranker": {"top_k": 1}}
    )

    print (prediction)


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
