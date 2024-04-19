import os
import faiss
import json
from abc import ABC, abstractmethod
from typing import cast, List, Union, Tuple
import warnings
import numpy as np
import shutil
import subprocess
import argparse
import torch
from tqdm import tqdm
from sqlite_utils import Database
from flashrag.retriever.utils import load_model, load_corpus, pooling, base_content_function

class Index_Builder:
    r"""A tool class used to build an index used in retrieval.
    
    """

    def __init__(
            self, 
            retrieval_method,
            model_path,
            corpus_path,
            save_dir,
            max_length,
            batch_size,
            use_fp16,
            pooling_method,
            save_corpus,
            faiss_type=None,
            save_embedding=False,
            content_function: callable = base_content_function
        ):
        
        self.retrieval_method = retrieval_method.lower()
        self.model_path = model_path
        self.corpus_path = corpus_path
        self.save_dir = save_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.pooling_method = pooling_method
        self.faiss_type = faiss_type if faiss_type else 'Flat'
        self.save_embedding = save_embedding
        self.save_corpus = save_corpus

        self.gpu_num = torch.cuda.device_count()
        # prepare save dir
        print(self.save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            if not self._check_dir(self.save_dir):
                warnings.warn(f"Some files already exists in {self.save_dir} and may be overwritten.", UserWarning)

        self.index_save_path = os.path.join(self.save_dir, f"{self.retrieval_method}_{self.faiss_type}.index")
        self.database_save_path = os.path.join(self.save_dir, "corpus.db")
        self.embedding_save_path = os.path.join(self.save_dir, "emb.memmap")

        self.corpus, self.have_contents, self.corpus_size = load_corpus(self.corpus_path, content_function)
        self.content_function = content_function

        # TODO: 在config模块完成后自动将路径存入yaml中

    @staticmethod
    def _check_dir(dir_path):
        r"""Check if the dir path exists and if there is content.
        
        """
        
        if os.path.isdir(dir_path):
            if len(os.listdir(dir_path)) > 0:
                return False
        else:
            os.makedirs(dir_path, exist_ok=True)
        return True

    def build_index(self):
        r"""Constructing different indexes based on selective retrieval method.

        """
        if self.retrieval_method == "bm25":
            self.build_bm25_index()
        else:
            self.build_dense_index()

    def build_bm25_index(self):
        """Building BM25 index based on Pyserini library.

        Reference: https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-bm25-index-direct-java-implementation
        """

        # to use pyserini pipeline, we first need to place jsonl file in the folder 
        self.save_dir = os.path.join(self.save_dir, "bm25")
        os.makedirs(self.save_dir,exist_ok=True)
        temp_dir = self.save_dir + "/temp"
        temp_file_path = temp_dir + "/temp.jsonl"
        os.makedirs(temp_dir)

        if self.have_contents:
            shutil.copyfile(self.corpus_path, temp_file_path)
        else:
            with open(temp_file_path, "w") as f:
                for item in self.corpus:
                    f.write(json.dumps(item) + "\n")
        
        print("Start building bm25 index...")
        pyserini_args = ["--collection", "JsonCollection",
                         "--input", temp_dir,
                         "--index", self.save_dir,
                         "--generator", "DefaultLuceneDocumentGenerator",
                         "--threads", "1"]
       
        subprocess.run(["python", "-m", "pyserini.index.lucene"] + pyserini_args)

        shutil.rmtree(temp_dir)
        
        print("Finish!")

    def _save_embedding(self, all_embeddings):
        memmap = np.memmap(
            self.embedding_save_path,
            shape=all_embeddings.shape,
            mode="w+",
            dtype=all_embeddings.dtype
        )
        length = all_embeddings.shape[0]
        # add in batch
        save_batch_size = 10000
        if length > save_batch_size:
            for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                j = min(i + save_batch_size, length)
                memmap[i: j] = all_embeddings[i: j]
        else:
            memmap[:] = all_embeddings

    @torch.no_grad()
    def build_dense_index(self):
        r"""Obtain the representation of documents based on the embedding model(BERT-based) and 
        construct a faiss index.
        
        """
        # TODO: disassembly overall process, use non open-source emebdding/ processed embedding
        # TODO: save embedding
        # prepare model
        if os.path.exists(self.index_save_path):
            print("The index file already exists and will be overwritten.")

        self.encoder, self.tokenizer = load_model(model_path = self.model_path, 
                                                  use_fp16 = self.use_fp16)
        if self.gpu_num > 1:
            print("Use multi gpu!")
            self.encoder = torch.nn.DataParallel(self.encoder)
            self.batch_size = self.batch_size * self.gpu_num


        all_embeddings = []

        for start_index in tqdm(range(0, self.corpus_size, self.batch_size), 
                                desc="Inference Embeddings"):
            sentences_batch = []
            for _ in range(self.batch_size):
                try:
                    sentence = next(self.corpus)['contents']
                    sentences_batch.append(sentence)
                except StopIteration:
                    break
            if not sentences_batch:
                print(f"stop at: {start_index}")
                break
            if self.retrieval_method == "e5":
                sentences_batch = [f"passage: {doc}" for doc in sentences_batch]
            inputs = self.tokenizer(
                        sentences_batch,
                        padding=True,
                        truncation=True,
                        return_tensors='pt',
                        max_length=self.max_length,
            ).to('cuda')

            inputs = {k: v.cuda() for k, v in inputs.items()}

            #TODO: support encoder-only T5 model
            if "T5" in type(self.encoder).__name__:
                # T5-based retrieval model
                decoder_input_ids = torch.zeros(
                    (inputs['input_ids'].shape[0], 1), dtype=torch.long
                ).to(inputs['input_ids'].device)
                output = self.encoder(
                    **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
                )
                embeddings = output.last_hidden_state[:, 0, :]

            else:
                output = self.encoder(**inputs, return_dict=True)
                embeddings = pooling(output.pooler_output, 
                                    output.last_hidden_state, 
                                    inputs['attention_mask'],
                                    self.pooling_method)
                if  "dpr" not in self.retrieval_method:
                    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

            embeddings = cast(torch.Tensor, embeddings)
            embeddings = embeddings.detach().cpu().numpy()
            all_embeddings.append(embeddings)


        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_embeddings = all_embeddings.astype(np.float32)
        if self.save_embedding:
            self._save_embedding(all_embeddings)

        # build index
        print("Creating index")
        dim = all_embeddings.shape[-1]
        faiss_index = faiss.index_factory(dim, self.faiss_type, faiss.METRIC_L2)
        
        # faiss_index.train(all_embeddings)
        # faiss_index.add(all_embeddings)

        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)
        faiss_index.train(all_embeddings)
        faiss_index.add(all_embeddings)
        faiss_index = faiss.index_gpu_to_cpu(faiss_index)

        faiss.write_index(faiss_index, self.index_save_path)
        

        if self.save_corpus:
            if os.path.exists(self.database_save_path):
                print("The database already exists and will not be written.") 
            else:
                print("Begin creating database...")
                # build corpus databse
                db = Database(self.database_save_path)
                docs = db['docs']
                docs.insert_all(self.corpus, batch_size=1000000, truncate=True)
                db.execute("CREATE INDEX idx_id ON docs (id)")
                db.conn.close()

        print("Finish!")



MODEL2POOLING = {
    "e5": "mean",
    "bge": "cls",
    "contriever": "mean"
}

def main():
    parser = argparse.ArgumentParser(description = "Creating index.")

    # Basic parameters
    parser.add_argument('--retrieval_method', type=str)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--corpus_path', type=str)
    parser.add_argument('--save_dir', default= 'indexes/',type=str)

    # Parameters for building dense index
    parser.add_argument('--max_length', type=int, default=180)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--use_fp16', default=False, action='store_true')
    parser.add_argument('--pooling_method', type=str, default=None)
    parser.add_argument('--faiss_type',default=None,type=str)
    parser.add_argument('--save_corpus', action='store_true',default=False)
    parser.add_argument('--save_embedding', action='store_true', default=False)
    
    args = parser.parse_args()

    if args.pooling_method is None:
        pooling_method = 'pooler'
        for k,v in MODEL2POOLING.items():
            if k in args.retrieval_method.lower():
                pooling_method = v
                break
    else:
        if args.pooling_method not in ['mean','cls','pooler']:
            raise NotImplementedError
        else:
            pooling_method = args.pooling_method


    index_builder = Index_Builder(
                        retrieval_method = args.retrieval_method,
                        model_path = args.model_path,
                        corpus_path = args.corpus_path,
                        save_dir = args.save_dir,
                        max_length = args.max_length,
                        batch_size = args.batch_size,
                        use_fp16 = args.use_fp16,
                        pooling_method = pooling_method,
                        save_corpus = args.save_corpus,
                        faiss_type = args.faiss_type,
                        save_embedding = args.save_embedding
                    )
    index_builder.build_index()


if __name__ == "__main__":
    main()