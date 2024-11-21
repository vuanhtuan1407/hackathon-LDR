import argparse
import json
import os

import pandas as pd
from tqdm import tqdm

MAX_NEG_SAMPLES = 1  # memory not allowed
NEG_POS = 42  # random number to get neg samples: pos = i => neg = [i-42, i-43, ..., i-42-MAX_NEG_SAMPLES]


# NEG_POS + MAX_NEG_SAMPLES < total samples

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_filepath', type=str, default='../../data/tmp_data/train_formatted.csv')
    parser.add_argument('--save_filepath', type=str, default='../../data/json_data/train.jsonl')
    return parser.parse_args()


def gen_train_bge(train_filepath='train_formatted.csv', save_filepath='train_bge.jsonl'):
    """Generate train samples for BGE based on Contrastive Learning"""
    df = pd.read_csv(train_filepath)
    questions = df['question'].tolist()
    contexts = df['context'].tolist()

    def chunk_ctx(ctx):
        """Chunk context into sentences because of limited length of BGE model and memory"""
        tmp = []
        ctx = eval(ctx)
        for c in ctx:
            c = c.split('\n')
            tmp.extend(c)
        return tmp

    p_contexts = [chunk_ctx(ctx) for ctx in contexts]

    assert len(questions) == len(contexts)
    n_samples = len(df)

    def gen_neg_samples(pos_idx, max_neg_samples=5):
        """Generate negative samples based on the position and max_neg_samples"""
        n_ctx = []
        for k in range(max_neg_samples):
            n_ctx.extend(p_contexts[pos_idx - NEG_POS - k])
        return n_ctx

    train_bge = []
    for i in tqdm(range(n_samples), total=n_samples, desc='Generating train BGE samples'):
        n_contexts = gen_neg_samples(i, MAX_NEG_SAMPLES)
        bge_sample = {
            "query": questions[i],
            'pos': p_contexts[i],
            'neg': n_contexts,
            'prompt': 'Represent this sentence for searching relevant passages related to the law:',
            'type': 'symmetric_sts'
        }

        train_bge.append(bge_sample)

    print(f'Saving train BGE samples to {save_filepath}')
    with open(save_filepath, 'w', encoding='utf-8') as f:
        json.dump(train_bge, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    args = parse_arguments()
    if not os.path.exists(args.train_filepath):
        raise FileNotFoundError(f'Train file {args.train_filepath} not found')

    save_dir = os.path.dirname(args.save_filepath)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    gen_train_bge(args.train_filepath, args.save_filepath)
