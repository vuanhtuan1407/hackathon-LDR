import argparse
import ast
import os
import re

import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_filepath', type=str, default='../../data/raw_data/train.csv')
    parser.add_argument('--save_filepath', type=str, default='../../data/tmp_data/train_formatted.csv')
    return parser.parse_args()


def clean_train_data(train_filepath='train.csv', save_filepath='train_formatted.csv'):
    """Format context string and convert cid to list[int]"""
    df = pd.read_csv(train_filepath)

    def clean_context(ctx):
        ctx = ctx.replace('\'', '"')
        ctx = re.sub(r'\\xad', '', ctx)
        ctx = re.sub(r'\\xa0', ' ', ctx)
        ctx = re.sub(r'[“”‘’]', '"', ctx)
        ctx = re.sub(r'"+', '"', ctx)
        ctx = re.sub(r'\\n"]|\["|"]', '', ctx)
        ctx = re.sub(r'(\d+\.\d+)\\n', r'\1. ', ctx)
        ctx = re.sub(r'\\n\[\.\.\.]|\\n\.\.\.', '', ctx)
        ctx = re.sub(r'\\n…', '', ctx)
        ctx = re.sub(r'\\n\.\.\.\\n', '\\n', ctx)
        ctx = re.sub(r'\s+', ' ', ctx)
        ctx = re.sub(r'\\n', '\n', ctx)  # \u000A = \n
        ctx = re.sub(r'"\s+"', '|', ctx)
        ctx = ctx.split('|')
        return ctx

    def format_cid(cid):
        cid = re.sub(r'\s*\[\s*', '[', cid)
        cid = re.sub(r'\s*]\s*', ']', cid)
        cid = re.sub(r'\s+', ',', cid)
        cid = ast.literal_eval(cid)
        return cid

    df['context'] = df['context'].apply(clean_context)
    df['cid'] = df['cid'].apply(format_cid)

    print(df['context'][525][0])

    print(f'Saving formatted train data to {save_filepath}')
    df.to_csv(save_filepath, index=False)


if __name__ == '__main__':
    args = parse_arguments()
    if not os.path.exists(args.train_filepath):
        raise FileNotFoundError(f'Train file {args.train_filepath} not found')

    save_dir = os.path.dirname(args.save_filepath)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    clean_train_data(args.train_filepath, args.save_filepath)
