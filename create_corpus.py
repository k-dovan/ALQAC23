import json
import os
from tqdm import tqdm
from alqac_utils import read_json_sets
from task1 import *

if __name__ == '__main__':

    corpora_data = read_json_sets(file_paths=[path for path in CORPORA.values()])

    save_dict = {}
    for law_article in tqdm(corpora_data):
        law_id = law_article["id"]
        law_articles = law_article["articles"]
        
        for sub_article in law_articles:
            article_id = sub_article["id"]
            article_text = sub_article["text"]
            
            concat_id = law_id + "_" + article_id
            if concat_id not in save_dict:
                save_dict[concat_id] = article_text
    
    save_path = f"{DATASET_DIR}/generated_finetuning_data"
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "corpus_dict.json"), 'w', encoding='utf-8') as f:
            json_object = json.dumps(save_dict, indent=4, ensure_ascii=False)
            f.write(json_object) 
