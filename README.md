# ALQAC23
Sonic team's repo for QLQAC 2023


## Run task 1:
### Retriever only
#### BM25 Retriever
python task1.py --corpus="ALQAC2023" --retrieval_method='BM25Retriever' --retriever_top_k=1

#### Td-idf Retriever
python task1.py --corpus="ALQAC2023" --retrieval_method='TfidfRetriever' --retriever_top_k=1

### With Ranker
#### BM25 Retriever
python task1.py --corpus="ALQAC2023" --retrieval_method='BM25Retriever' --retriever_top_k=50 --ranker_top_k=1 --with_ranker

#### Td-idf Retriever
python task1.py --corpus="ALQAC2023" --retrieval_method='TfidfRetriever' --retriever_top_k=50 --ranker_top_k=1 --with_ranker