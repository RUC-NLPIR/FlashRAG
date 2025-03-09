CUDA_VISIBLE_DEVICES=0 python -m flashrag.retriever.index_builder \
    --retrieval_method openai-clip-vit-large-patch14-336 \
    --model_path openai/clip-vit-large-patch14-336 \
    --corpus_path datasets/mmqa/train.parquet \
    --save_dir indexes/mmqa \
    --max_length 512 \
    --batch_size 512 \
    --faiss_type Flat \
    --index_modal all


# CUDA_VISIBLE_DEVICES=0 python -m flashrag.retriever.index_builder \
#     --retrieval_method bm25 \
#     --corpus_path datasets/mmqa/train.parquet \
#     --save_dir indexes/mmqa \
#     --max_length 512 \
#     --batch_size 512 \
#     --bm25_backend bm25s
