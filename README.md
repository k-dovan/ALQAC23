# ALQAC23
Sonic team's repo for QLQAC 2023


## Run task 1:
### Retriever only
#### BM25 Retriever
python task1.py --corpus="ALQAC2023" --retrieval_method='BM25Retriever' --retriever_top_k=1 --print_coverage

##### Experiments(k=retriever_top_k, cov=coverage): 
##### [k=5,cov=93%; k=6,cov=94%; k=7,cov=95%; k=8-15,cov=97%; k=16-36,cov=98%; k=37-48,cov=99%; k=49-,cov=100%]

#### Td-idf Retriever
python task1.py --corpus="ALQAC2023" --retrieval_method='TfidfRetriever' --retriever_top_k=1 --print_coverage

##### Experiments(k=retriever_top_k, cov=coverage): 
##### [k=8-10,cov=97%; k=11,cov=98%; k=12-35,cov=99%; k=36-,cov=100%]

### With Ranker
#### BM25 Retriever
python task1.py --corpus="ALQAC2023" --retrieval_method='BM25Retriever' --retriever_top_k=50 --ranker_top_k=1 --with_ranker --print_coverage --ranker_model='cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'

#### Td-idf Retriever
python task1.py --corpus="ALQAC2023" --retrieval_method='TfidfRetriever' --retriever_top_k=50 --ranker_top_k=1 --with_ranker --print_coverage --ranker_model='cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'

## Experiments
### Bm25TRetriever with ranker
- python task1.py --corpus="ALQAC2023" --retrieval_method='BM25Retriever' --retriever_top_k=[20,30] --ranker_top_k=1 --with_ranker --print_coverage --ranker_model='cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
- Gained: 84% coverage.


### TdidfRetriever with ranker
- python task1.py --corpus="ALQAC2023" --retrieval_method='TfidfRetriever' --retriever_top_k=[20,30] --ranker_top_k=1 --with_ranker --print_coverage --ranker_model='cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
- Gained: % coverage.

## Prepare finetuning data pairs
### with Bm25TRetriever
- python prepare_finetuning_data.py --retrieval_method='BM25Retriever' --retriever_top_k=200

### with TdidfRetriever
- python prepare_finetuning_data.py --retrieval_method='TfidfRetriever' --retriever_top_k=200