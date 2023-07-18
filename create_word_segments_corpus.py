from alqac_utils import bm25_tokenizer, read_json_sets
from underthesea import word_tokenize
from tqdm import tqdm


DATASET_DIR = "../ALQAC_2023_training_data"
SAVED_PATH = f'{DATASET_DIR}/cleaned_corpus/'

CORPUS_FILE = f'{DATASET_DIR}/law.json'

corpus_dict = read_json_sets([CORPUS_FILE])

word_segments_set = set()

for law in tqdm(corpus_dict):
    for art in law["articles"]:
        if not ("text" in art.keys()):
            continue
        text = art["text"].replace("\n\n", " ")
        for w in word_tokenize(text):
            word_segments_set.add(w)

with open(f"{SAVED_PATH}/cleaned_word_segments_corpus.txt", 'w', encoding="utf-8") as f:
    for w in tqdm(word_segments_set):
        f.write("%s\n" % w)

with open(f"{SAVED_PATH}/cleaned_1_word_corpus.txt", 'w', encoding="utf-8") as f:
    for w in tqdm(word_segments_set):
        if " " not in w:
            f.write("%s\n" % w)
        