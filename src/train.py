from FlagEmbedding import BGEM3FlagModel as BGEM3

model = BGEM3(
    model_name_or_path='BAAI/bge-m3',
    use_fp16=True,
    device='cuda',
    query_instruction_for_retrieval='Represent this sentence for searching relevant passages related to the law:'
)

print(model)