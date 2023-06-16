import json
from typing import List

from haystack.document_stores import InMemoryDocumentStore

# ============================= data utils ====================================
# read law data from a json file to dict (with required format)
# {"content": "...", "meta": {"law_id": "05/2022/QH15", "article_id": "95"}}


def read_corpus(file_path: str) -> List[dict]:
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
                item = {"content": art["text"], "meta": {
                    "law_id": law["id"], "article_id": art["id"]}}
                res.append(item)

    return res


def prepare_in_memory_dataset(file_paths: List[str]) -> InMemoryDocumentStore:
    document_store = InMemoryDocumentStore(
        use_bm25=True,
        similarity='dot_product',
        embedding_dim=768
    )
    # load data from json files
    data = []
    for file_path in file_paths:
        data.extend(read_corpus(file_path))

    # skip duplicate documents if exist
    document_store.write_documents(
        data, batch_size=1000, duplicate_documents="skip")

    return document_store


def read_eval_sets(file_paths: List[str]) -> List[dict]:
    res = []
    for file_path in file_paths:
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                continue
            res.extend(data)
    return res

# ================================ metrics =========================================
