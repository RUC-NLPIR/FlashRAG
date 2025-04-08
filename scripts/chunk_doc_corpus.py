"""Chunk documents from a Document Corpus JSONL file.

This script is used to chunk documents from a Document Corpus JSONL file,
via Chonkie.
"""

import argparse
import json
from tqdm import tqdm
import chonkie


def load_jsonl(file_path):
    documents = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))
    return documents


def save_jsonl(documents, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk documents from a JSONL file.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument(
        "--chunk_by", default="token", choices=["token", "word", "sentence", "recursive"], help="Chunking method to use"
    )
    parser.add_argument("--chunk_size", default=512, type=int, help="Size of chunks")
    parser.add_argument("--tokenizer_name_or_path", default="o200k_base", type=str)

    args = parser.parse_args()

    # Load documents
    print("Loading documents...")
    documents = load_jsonl(args.input_path)

    # Initialize chunker
    if args.chunk_by == "token":
        chunker = chonkie.TokenChunker(tokenizer=args.tokenizer_name_or_path, chunk_size=args.chunk_size)
    elif args.chunk_by == "word":
        chunker = chonkie.TokenChunker(tokenizer="word", chunk_size=args.chunk_size)
    elif args.chunk_by == "sentence":
        chunker = chonkie.SentenceChunker(tokenizer_or_token_counter=args.tokenizer_name_or_path, chunk_size=args.chunk_size)
    elif args.chunk_by == "recursive":
        chunker = chonkie.RecursiveChunker(
            tokenizer_or_token_counter=args.tokenizer_name_or_path, chunk_size=args.chunk_size, min_characters_per_chunk=1
        )
    else:
        raise ValueError(f"Invalid chunking method: {args.chunk_by}")

    # Process and chunk documents
    print("Chunking documents...")
    chunked_documents = []
    current_chunk_id = 0
    for doc in tqdm(documents):
        title, text = doc["contents"].split("\n", 1)
        chunks = chunker.chunk(text)
        for chunk in chunks:
            chunked_doc = {
                "id": current_chunk_id,
                "doc_id": doc["id"],
                "title": title,
                "contents": title + "\n" + chunk.text,
            }
            chunked_documents.append(chunked_doc)
            current_chunk_id += 1

    # Save chunked documents
    print("Saving chunked documents...")
    save_jsonl(chunked_documents, args.output_path)
    print(f"Done! Processed {len(documents)} documents into {len(chunked_documents)} chunks.")
