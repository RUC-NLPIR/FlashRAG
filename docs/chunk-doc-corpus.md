# Chunking Document Corpus

You can chunk your document corpus into smaller chunks by following these steps. This is useful in building an index over a large corpus of long documents for RAG, or if you want to make sure that the document length is not too long for the model.

Given a JSONL file with the following format:

```json
{
    "id": "doc_id",
    "content": "document_content"
}
```

run the following command:

```bash
cd scripts
python chunk_doc_corpus.py --input_path input.jsonl \
                          --output_path output.jsonl \
                          --chunk_by sentence \
                          --chunk_size 512
```

And you will get a JSONL file with the following format:

```json
{
    "id": "doc_id",
    "content": "document_content"
}
```

**NOTE:** That `doc_id` will be the same as the original document id, and `content` will be the chunked document content in the new JSONL output.

## Parameters

- `input_path`: Path to the input JSONL file.
- `output_path`: Path to the output JSONL file.
- `chunk_by`: Chunking method to use. Can be `token`, `word`, `sentence`, or `recursive`.
- `chunk_size`: Size of chunks.
