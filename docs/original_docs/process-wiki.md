# Building Wiki Corpus

You can create your own wiki corpus by following these steps.

## Step1: Download Wiki dump

Download the Wikipedia dump you require in XML format. For instance: 

```bash
wget https://archive.org/download/enwiki-20181220/enwiki-20181220-pages-articles.xml.bz2
```

You can access other dumps from this [<u>website</u>](https://archive.org/search?query=Wikimedia+database+dump&sort=-downloads).

## Step2: Run process script

Execute the provided script to process the wiki dump into JSONL format. Adjust the corpus partitioning parameters as needed:

```bash
cd scripts
python preprocess_wiki.py --dump_path ../enwikinews-20240420-pages-articles.xml.bz2  \
                        --save_path ../test_sample.jsonl \
                        --chunk_by sentence \
                        --seg_size 6 \
                        --stride 1 \
                        --num_workers 1
```


We also provide the version we used for experiments. Download link: https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/main/retrieval-corpus
