import os

import pandas as pd
from FlagEmbedding import BGEM3FlagModel as BGEM3
import faiss
import numpy as np

CORPUS_PATH = '../data/raw_data/corpus.csv'
TEST_PATH = '../data/raw_data/public_test.csv'
INDEX_PATH = '../data/tmp_data/corpus_index.bin'
RESULT_PATH = '../data/tmp_data/result.txt'

NUM_TEST = 1000

model = BGEM3(
    model_name_or_path='BAAI/bge-m3',
    use_fp16=True,
    device='cuda',
    query_instruction_for_retrieval='Represent this sentence for searching relevant passages related to the law:'
)


def emb_corpus():
    df = pd.read_csv(CORPUS_PATH)
    corpus = df['text'].tolist()[:NUM_TEST]
    cid = df['cid'].to_numpy()[:NUM_TEST]
    embeddings = model.encode(corpus, convert_to_numpy=True)['dense_vecs']
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
    else:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, INDEX_PATH)

    return cid, index


def infer():
    cid, index = emb_corpus()
    df = pd.read_csv(TEST_PATH)
    queries = df['question'].tolist()[:3]
    qid = df['qid'].tolist()[:3]
    q_embeddings = model.encode(queries, convert_to_numpy=True)['dense_vecs']
    _, I = index.search(q_embeddings, 10)
    res = []
    for i in range(len(queries)):
        print(f'Question: {queries[i]}')
        print(f'Question ID: {qid[i]}')
        print(f'Relevant Passages: {cid[I[i]]}')
        print('--------------------------------')
        res.append(np.concat([[qid[i]], cid[I[i]]], axis=-1))

    print(f'Save result to {RESULT_PATH}')
    np.savetxt(RESULT_PATH, res, fmt='%d')


if __name__ == '__main__':
    infer()
