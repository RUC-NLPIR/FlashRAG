# Chunking Document Corpus

You can chunk your document corpus into smaller chunks by following these steps. This is useful in building an index over a large corpus of long documents for RAG, or if you want to make sure that the document length is not too long for the model.

Given a Document Corpus JSONL file with the following format and `contents` field containing the `"{title}\n{text}"` format:

```jsonl
{ "id": 0, "contents": "..." }
{ "id": 1, "contents": "..." }
{ "id": 2, "contents": "..." }
...
```

You can run the following command:

```bash
cd scripts
python chunk_doc_corpus.py --input_path input.jsonl \
                          --output_path output.jsonl \
                          --chunk_by sentence \
                          --chunk_size 512
```

You will get a JSONL file with the following format:

```jsonl
{ "id": 0, "doc_id": 0, "title": ..., "contents": ... }
{ "id": 1, "doc_id": 0, "title": ..., "contents": ... }
{ "id": 2, "doc_id": 0, "title": ..., "contents": ... }
...
```

**NOTE:** That `doc_id` will be the same as the original document id, and `contents` will be the chunked document content in the new JSONL output.

## Parameters

- `input_path`: Path to the input JSONL file.
- `output_path`: Path to the output JSONL file.
- `chunk_by`: Chunking method to use. Can be `token`, `word`, `sentence`, or `recursive`.
- `chunk_size`: Size of chunks.
- `tokenizer_name_or_path`: Name or path of the tokenizer that used for chunking.
