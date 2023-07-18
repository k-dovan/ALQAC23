from alqac_utils import bm25_tokenizer, read_json_sets
from tqdm import tqdm
import json


DATASET_DIR = "ALQAC_2023_training_data"
SAVED_PATH = f'{DATASET_DIR}/cleaned_corpus/'

CORPUS_FILES = [
                f'{DATASET_DIR}/law.json', 
                f'{DATASET_DIR}/additional_data/ALQAC_2022_training_data/law.json', 
                # f'{DATASET_DIR}/additional_data/zalo/zalo_corpus.json'
            ]
CLEANED_SAVED_FILES = [
                    f'{SAVED_PATH}/law_2023_cleaned.json', 
                    f'{SAVED_PATH}/law_2022_cleaned.json', 
                    #    f'{SAVED_PATH}/law_zalo_cleaned.json'
                    ]

for i, corpus_file in enumerate(CORPUS_FILES):
    res = []
    with open(corpus_file, "r", encoding='utf-8') as f:
        data = json.load(f)
        if not isinstance(data, list):
            continue
        for law in tqdm(data):
            if not ("id" in law.keys() and "articles" in law.keys()):
                continue
            if not isinstance(law["articles"], list):
                continue
            for art in law["articles"]:
                if not ("id" in art.keys() and "text" in art.keys()):
                    continue
                
                # clean article text
                cleaned = ' '.join(bm25_tokenizer(art["text"]))
                item = {"content": cleaned, "meta": {
                    "law_id": law["id"], "article_id": art["id"]}}
                res.append(item)
    with open(CLEANED_SAVED_FILES[i], 'w', encoding="utf-8") as f:
        json_object = json.dumps(res, indent=4, ensure_ascii=False)
        f.write(json_object)
        
