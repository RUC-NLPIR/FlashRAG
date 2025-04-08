import argparse
from tqdm import tqdm
import re
import html
import os
import json
import subprocess
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool


def load_corpus(dir_path):
    def iter_files(path):
        """Walk through all files located under a root path."""
        if os.path.isfile(path):
            yield path
        elif os.path.isdir(path):
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    yield os.path.join(dirpath, f)
        else:
            raise RuntimeError("Path %s is invalid" % path)

    def read_jsonl_file(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                json_data = json.loads(line)
                corpus.append(json_data)

    all_files = [file for file in iter_files(dir_path)]
    corpus = []

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        for file_path in all_files:
            executor.submit(read_jsonl_file, file_path)

    return corpus


def basic_process(title, text):
    title = html.unescape(title)
    text = html.unescape(text)
    text = text.strip()

    if "(disambiguation)" in title.lower():
        return None, None
    if "(disambiguation page)" in title.lower():
        return None, None
    # Take out List/Index/Outline pages (mostly links)
    if re.match(r"(List of .+)|(Index of .+)|(Outline of .+)", title):
        return None, None
    if text.startswith("REDIRECT") or text.startswith("redirect"):
        return None, None
    if text.endswith(". References."):
        text = text[: -len(" References.")].strip()

    text = re.sub("\{\{cite .*?\}\}", " ", text, flags=re.DOTALL)
    text = text.replace(r"TABLETOREPLACE", " ")
    text = text.replace(r"'''", " ")
    text = text.replace(r"[[", " ")
    text = text.replace(r"]]", " ")
    text = text.replace(r"{{", " ")
    text = text.replace(r"}}", " ")
    text = text.replace("<br>", " ")
    text = text.replace("&quot;", '"')
    text = text.replace("&amp;", "&")
    text = text.replace("& amp;", "&")
    text = text.replace("nbsp;", " ")
    text = text.replace("formatnum:", "")

    # text = re.sub('<poem.*?</poem>', ' ', text, flags=re.DOTALL) # might have useful information?
    text = re.sub("<math.*?</math>", "", text, flags=re.DOTALL)
    text = re.sub("<chem.*?</chem>", "", text, flags=re.DOTALL)
    text = re.sub("<score.*?</score>", "", text, flags=re.DOTALL)

    # clean residual mess from xml dump that shouldn't have made its way here
    text = re.sub("\| ?item[0-9]?_?style= ?.*? ", " ", text)
    text = re.sub("\| ?col[0-9]?_?style= ?.*? ", " ", text)
    text = re.sub("\| ?row[0-9]?_?style= ?.*? ", " ", text)
    text = re.sub("\| ?style= ?.*? ", " ", text)
    text = re.sub("\| ?bodystyle= ?.*? ", " ", text)
    text = re.sub("\| ?frame_?style= ?.*? ", " ", text)
    text = re.sub("\| ?data_?style= ?.*? ", " ", text)
    text = re.sub("\| ?label_?style= ?.*? ", " ", text)
    text = re.sub("\| ?headerstyle= ?.*? ", " ", text)
    text = re.sub("\| ?list_?style= ?.*? ", " ", text)
    text = re.sub("\| ?title_?style= ?.*? ", " ", text)
    text = re.sub("\| ?ul_?style= ?.*? ", " ", text)
    text = re.sub("\| ?li_?style= ?.*? ", " ", text)
    text = re.sub("\| ?border-style= ?.*? ", " ", text)
    text = re.sub('\|? ?style=".*?"', "", text)
    text = re.sub('\|? ?rowspan=".*?"', "", text)
    text = re.sub('\|? ?colspan=".*?"', "", text)
    text = re.sub('\|? ?scope=".*?"', "", text)
    text = re.sub('\|? ?align=".*?"', "", text)
    text = re.sub('\|? ?valign=".*?"', "", text)
    text = re.sub('\|? ?lang=".*?"', "", text)
    text = re.sub('\|? ?bgcolor=".*?"', "", text)
    text = re.sub("\|? ?bg=\#[a-z]+", "", text)
    text = re.sub('\|? ?width=".*?"', "", text)
    text = re.sub("\|? ?height=[0-9]+", "", text)
    text = re.sub("\|? ?width=[0-9]+", "", text)
    text = re.sub("\|? ?rowspan=[0-9]+", "", text)
    text = re.sub("\|? ?colspan=[0-9]+", "", text)
    text = re.sub(r"[\n\t]", " ", text)
    text = re.sub("<.*?/>", "", text)
    text = re.sub("\|? ?align=[a-z]+", "", text)
    text = re.sub("\|? ?valign=[a-z]+", "", text)
    text = re.sub("\|? ?scope=[a-z]+", "", text)
    text = re.sub("&lt;ref&gt;.*?&lt;/ref&gt;", " ", text)
    text = re.sub("&lt;.*?&gt;", " ", text)
    text = re.sub("File:[A-Za-z0-9 ]+\.[a-z]{3,4}(\|[0-9]+px)?", "", text)
    text = re.sub("Source: \[.*?\]", "", text)
    text = text.replace("Country flag|", "country:")
    text = text.replace("flag|", "country:")
    text = text.replace("flagicon|", "country:")
    text = text.replace("flagcountry|", "country:")
    text = text.replace("Flagu|", "country:")
    text = text.replace("display=inline", "")
    text = text.replace("display=it", "")
    text = text.replace("abbr=on", "")
    text = text.replace("disp=table", "")

    title = title.replace("\n", " ").replace("\t", " ")

    return title, text


def split_list(lst, n):
    """Split a list into n roughly equal parts."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def single_worker(docs):
    results = []
    for item in tqdm(docs):
        title, text = basic_process(item[0], item[1])
        if title is None:
            continue
        title = f'"{title}"'
        results.append((title, text))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate clean wiki corpus file for indexing.")
    parser.add_argument("--dump_path", type=str)
    parser.add_argument(
        "--use_chonkie",
        type=bool,
        default=True,
        action="store_true",
    )
    parser.add_argument("--chunk_by", default="token", choices=["token", "sentence", "recursive", "100w"], type=str)
    parser.add_argument("--chunk_size", default=512, type=int)
    parser.add_argument("--tokenizer_name_or_path", default="o200k_base", type=str)
    parser.add_argument("--seg_size", default=None, type=int)
    parser.add_argument("--stride", default=None, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--save_path", type=str, default="clean_corpus.jsonl")
    args = parser.parse_args()

    if args.use_chonkie:
        import chonkie
    else:
        assert args.chunk_by in ["100w", "sentence"], "Only supports sentence and 100w chunking without chonkie!"
        import spacy

        nlp = spacy.load("en_core_web_lg")

    # extract wiki dump
    temp_dir = os.path.join(Path(args.save_path).parent, "temp")
    os.makedirs(temp_dir)
    subprocess.run(
        [
            "python",
            "-m",
            "wikiextractor.WikiExtractor",
            "--json",
            "--filter_disambig_pages",
            "--quiet",
            "-o",
            temp_dir,
            "--process",
            str(args.num_workers),
            args.dump_path,
        ]
    )

    corpus = load_corpus(temp_dir)

    documents = {}
    # To avoid duplicate pages
    for item in tqdm(corpus):
        title = item["title"]
        text = item["text"]
        if title in documents:
            documents[title] += " " + text
        else:
            documents[title] = text

    print("Start pre-processing...")
    documents = list(documents.items())

    with Pool(processes=args.num_workers) as p:
        result_list = list(tqdm(p.imap(single_worker, split_list(documents, args.num_workers))))
    result_list = sum(result_list, [])

    all_title = [item[0] for item in result_list]
    all_text = [item[1] for item in result_list]

    print("Start chunking...")
    idx = 0
    clean_corpus = []

    if args.use_chonkie:
        print("Using Chonkie chunker...")
    else:
        print("Using default chunker...")

    if args.use_chonkie:
        # Initialize a Chonkie chunker, based on the chunk_by argument
        if args.chunk_by == "token":
            chunker = chonkie.TokenChunker(tokenizer=args.tokenizer_name_or_path, chunk_size=args.chunk_size)
        elif args.chunk_by == "sentence":
            chunker = chonkie.SentenceChunker(tokenizer_or_token_counter=args.tokenizer_name_or_path, chunk_size=args.chunk_size)
        elif args.chunk_by == "recursive":
            chunker = chonkie.RecursiveChunker(
                tokenizer_or_token_counter=args.tokenizer_name_or_path, chunk_size=args.chunk_size, min_characters_per_chunk=1
            )
        elif args.chunk_by == "100w":
            chunker = chonkie.TokenChunker(tokenizer="word", chunk_size=100)
        else:
            raise ValueError(f"Invalid chunking method: {args.chunk_by}")

        # Chunk the text into segments, with chunker
        for title, text in tqdm(zip(all_title, all_text), total=len(all_text)):
            chunks = chunker.chunk(text)
            for chunk in chunks:
                clean_corpus.append({"title": title, "text": chunk.text})
    else:
        if args.chunk_by == "sentence":
            for doc in tqdm(nlp.pipe(all_text, n_process=args.num_workers, batch_size=2000), total=len(all_text)):
                title = all_title[idx]
                idx += 1
                sentences = [sent.text.strip() for sent in doc.sents]
                segments = []
                for i in range(0, len(sentences), args.stride):
                    segment = " ".join(sentences[i : i + args.seg_size])
                    segments.append(segment)
                    if i + args.seg_size >= len(sentences):
                        break
                for segment in segments:
                    text = segment.replace("\n", " ").replace("\t", " ")
                    clean_corpus.append({"title": title, "text": text})

        elif args.chunk_by == "100w":
            for doc in tqdm(nlp.pipe(all_text, n_process=args.num_workers, batch_size=2000), total=len(all_text)):
                title = all_title[idx]
                idx += 1
                segments = []
                word_count = 0
                segment_tokens = []
                for token in doc:
                    segment_tokens.append(token.text_with_ws)
                    if not token.is_space and not token.is_punct:
                        word_count += 1
                        if word_count == 100:
                            word_count = 0
                            segments.append("".join([token for token in segment_tokens]))
                            segment_tokens = []
                if word_count != 0:
                    for token in doc:
                        segment_tokens.append(token.text_with_ws)
                        if not token.is_space and not token.is_punct:
                            word_count += 1
                            if word_count == 100:
                                word_count = 0
                                segments.append("".join([token for token in segment_tokens]))
                                break
                if word_count != 0:
                    segments.append("".join([token for token in segment_tokens]))

                for segment in segments:
                    text = segment.replace("\n", " ").replace("\t", " ")
                    clean_corpus.append({"title": title, "text": text})

    shutil.rmtree(temp_dir)

    print("Start saving corpus...")
    with open(args.save_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(clean_corpus):
            title = f"\"{item['title']}\""
            item = {"id": idx, "title": title, "text": item["text"]}
            f.write(json.dumps(item) + "\n")
    print("Finish!")
