## Building Wiki Corpus

You can create your own wiki corpus by following these steps.

### Step1: Install necessary tools

First, install the required tools:
```bash
pip install wikiextractor==0.1
python -m spacy download en_core_web_lg
```

If you encounter issues downloading en_core_web_lg, you can manually download it:
```bash
wget https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl
pip install en_core_web_lg-3.7.1-py3-none-any.whl
```


### Step2: Download Wiki dump

Download the Wikipedia dump you require in XML format. For instance: 

```bash
wget https://archive.org/download/enwiki-20181220/enwiki-20181220-pages-articles.xml.bz2
```

You can access other dumps from this [<u>website</u>](https://archive.org/search?query=Wikimedia+database+dump&sort=-downloads).


### Step3: Run process script

Execute the provided script to process the wiki dump into JSONL format. Adjust the corpus partitioning parameters as needed:

```bash
cd scripts
python preprocess_wiki.py --dump_path ../enwikinews-20240420-pages-articles.xml.bz2  \
                        --save_path ../test_sample.jsonl \
                        --chunk_by 100w
```

We also provide the version we used for experiments. Download link: https://huggingface.co/datasets/ignore/FlashRAG_datasets/tree/main/retrieval-corpus
