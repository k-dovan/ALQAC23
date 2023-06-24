"""
This examples show how to train a Cross-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).

In this example we use a knowledge distillation setup. Sebastian Hofstätter et al. trained in https://arxiv.org/abs/2010.02666
an ensemble of large Transformer models for the MS MARCO datasets and combines the scores from a BERT-base, BERT-large, and ALBERT-large model.

We use the logits scores from the ensemble to train a smaller model. We found that the MiniLM model gives the best performance while
offering the highest speed.

The resulting Cross-Encoder can then be used for passage re-ranking: You retrieve for example 100 passages
for a given query, for example with ElasticSearch, and pass the query+retrieved_passage to the CrossEncoder
for scoring. You sort the results then according to the output of the CrossEncoder.

This gives a significant boost compared to out-of-the-box ElasticSearch / BM25 ranking.

Running this script:
python train_cross-encoder-v2.py
"""
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers import InputExample
import logging
import argparse
import json
from random import random
import torch

from task1 import DATASET_DIR
    
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", default="mmarco-mMiniLMv2-L12-H384-v1", type=str, help="name of pretrained cross-encoder model")
    parser.add_argument("--max_seq_length", default=512, type=int, help="maximum sequence length")
    parser.add_argument("--data_type", type=str, default="bm25", help="finetuning data with `data_type` sampling method", choices=["bm25", "tfidf"])
    parser.add_argument("--validation_ratio_target", type=float, default=0.2, help="target percentage of validation samples in total")
    parser.add_argument("--evaluation_steps", default=5000, type=int, help="evaluate model after `evaluation_steps` of training steps")
    parser.add_argument("--epochs", default=5, type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=7e-6, help="learning rate for training")
    args = parser.parse_args()

    logging.info(f"Input args: {args}")

    #First, we define the transformer model we want to fine-tune
    # model_name = 'microsoft/MiniLM-L12-H384-uncased'  # original base model
    model_name = f'cross-encoder/{args.pretrained_model}'
    max_seq_length = args.max_seq_length
    ds_sampling_method = args.data_type     # the method that being used to sample the finetuning dataset
    validation_ratio_target = args.validation_ratio_target
    evaluation_steps = args.evaluation_steps
    num_epochs = args.epochs
    train_batch_size = args.batch_size
    learning_rate = args.lr

    datapairs_path = f"{DATASET_DIR}/generated_finetuning_data/query_doc_pairs_{ds_sampling_method}_top200.json"
    # path to save finetuned model
    model_save_path = f'saved_models/{args.pretrained_model}-VN-LegalQA-{ds_sampling_method}'

    #We set num_labels=1 and set the activation function to Identiy, so that we get the raw logits
    model = CrossEncoder(model_name, num_labels=1, max_length=max_seq_length, default_activation_function=torch.nn.Identity())

    ### Now we create our dev data
    train_samples = []    
    dev_samples = {}

    save_pairs = json.load(open(datapairs_path))
    logging.info(f"There are {len(save_pairs)} query-doc pairs.")
 
    for pair in save_pairs:
        question = pair["question"]
        document = pair["document"]
        relevant = float(pair["relevant"])

        sample = InputExample(texts=[question, document], label=relevant)

        if random() > validation_ratio_target:
            train_samples.append(sample)
        else:
            if question not in dev_samples.keys():
                dev_samples[question] = {'query': question, 'positive': set(), 'negative': set()}
            else:
                if relevant == 1:
                    dev_samples[question]['positive'].add(document)
                else:                    
                    dev_samples[question]['negative'].add(document)

    logging.info(f"Number of sample for training: {len(train_samples)}")

    # We create a DataLoader to load our train samples
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)

    # We add an evaluator, which evaluates the performance during training
    # It performs a classification task and measures scores like F1 (finding relevant passages) and Average Precision
    evaluator = CERerankingEvaluator(dev_samples, name='train-eval')

    # Configure the training
    warmup_steps = 5000
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_dataloader=train_dataloader,
            loss_fct=torch.nn.MSELoss(),
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=evaluation_steps,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            optimizer_params={'lr': learning_rate},
            use_amp=True)

    #Save latest model
    model.save(model_save_path+'-latest')