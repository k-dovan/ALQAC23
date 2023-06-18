from alqac_utils import *
from task1 import *


if __name__ == "__main__":

    # 1. prepare corpus and eval sets
    CORPUS = [f'{DATASET_DIR}/law.json',
                    #   f'{DATASET_DIR}/additional_data/ALQAC_2022_training_data/law.json',
                    #   f'{DATASET_DIR}/additional_data/zalo/zalo_corpus.json'
                    ]
    eval_paths = [f'{DATASET_DIR}/train.json',
                  #   f'{DATASET_DIR}/additional_data/ALQAC_2022_training_data/question.json',
                  #   f'{DATASET_DIR}/additional_data/zalo/zalo_question.json'
                  ]

    # load corpus datasets
    document_store = prepare_in_memory_dataset(file_paths=CORPUS)

    # load eval sets
    eval_sets = read_json_sets(eval_paths)

    # evaluate pipelines
    retrieval_method = RETRIEVAL_METHODS.BM25Retriever
    
    # build retriever
    retriever = build_retriever(document_store=document_store, retrieval_method=retrieval_method)

    # build retriver pipeline without Ranker
    pipeline = build_retriever_pipe(retriever=retriever, retrival_method=retrieval_method.name)

    # evaluate pipeline with provided F2-metric
    Precision, Recall, F2 = evaluate_pipeline(eval_sets=eval_sets, 
                                              pipeline=pipeline, 
                                              retrival_method=retrieval_method.name,
                                              retriever_top_k=1
                                              )

    print (f"Precision: {Precision}, Recall: {Recall}, F2: {F2}")

    
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