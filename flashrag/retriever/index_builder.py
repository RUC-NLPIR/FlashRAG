import os
from flashrag.retriever.utils import load_model, load_corpus, pooling, base_content_function, load_database
from flashrag.config.config import Config
import faiss
import json
from abc import ABC, abstractmethod
from typing import cast, List, Union, Tuple
import warnings
import numpy as np
import argparse
import torch
import shutil
import subprocess
from tqdm import tqdm
from sqlite_utils import Database

class Index_Builder:
    r"""A tool class used to build an index used in retrieval.
    
    """

    def __init__(
            self, 
            config,
            content_function: callable = base_content_function
        ):
        
        self.retrieval_method = config['retrieval_method']
        self.corpus_path = config['corpus_path']
        self.database_path = config['corpus_database_save_path']
        self.save_dir = config['index_save_dir']
        self.device = config['device']
        self.retrieval_model_path = config['retrieval_model_path']
        self.use_fp16 = config['index_use_fp16']
        self.batch_size = config['index_batch_size']
        self.pooling_method = config['retrieval_pooling_method']
        self.doc_max_length = config['index_doc_max_length']

        # prepare save dir
        self.save_dir = os.path.join(self.save_dir, self.retrieval_method)
        print(self.save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            if not self._check_dir(self.save_dir):
                warnings.warn(f"Some files already exists in {self.save_dir} and may be overwritten.", UserWarning)

        self.corpus = load_corpus(self.corpus_path)
        self.content_function = content_function

        self.have_contents = 'contents' in self.corpus[0]       
        
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
        temp_dir = self.save_dir + "/temp"
        temp_file_path = temp_dir + "/temp.jsonl"
        os.makedirs(temp_dir)

        if self.have_contents:
            shutil.copyfile(self.corpus_path, temp_file_path)
        else:
            with open(temp_file_path, "w") as f:
                for item in self.corpus:
                    item = json.loads(item)
                    item['contents'] = self.content_function(item)
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

    @torch.no_grad()
    def build_dense_index(self):
        r"""Obtain the representation of documents based on the embedding model(BERT-based) and 
        construct a faiss index.
        
        """
        import torch
        self.gpu_num = torch.cuda.device_count()
        # TODO: disassembly overall process, use non open-source emebdding/ processed embedding
        # TODO: save embedding
        # prepare model
        self.encoder, self.tokenizer = load_model(model_path = self.retrieval_model_path, 
                                                  use_fp16 = self.use_fp16)
        self.encoder.to(self.device)
        if self.gpu_num > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)
            self.batch_size = self.batch_size * self.gpu_num

        # get embeddings
        doc_content = [item['contents'] for item in self.corpus]

        if self.retrieval_method == "e5":
            doc_content = [f"passage: {doc}" for doc in doc_content]

        all_embeddings = []

        for start_index in tqdm(range(0, len(doc_content), self.batch_size), 
                                desc="Inference Embeddings",
                                disable=len(doc_content) < self.batch_size):
            sentences_batch = doc_content[start_index:start_index + self.batch_size]
            inputs = self.tokenizer(
                        sentences_batch,
                        padding=True,
                        truncation=True,
                        return_tensors='pt',
                        max_length=self.doc_max_length,
            ).to(self.device)
            output = self.encoder(**inputs, return_dict=True)
            embeddings = pooling(output.pooler_output, 
                                 output.last_hidden_state, 
                                 inputs['attention_mask'],
                                 self.pooling_method)

            embeddings = cast(torch.Tensor, embeddings)
            if "dpr" not in self.retrieval_method:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1).detach()

            embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)


        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_embeddings = all_embeddings.astype(np.float32)

        # build index
        dim = all_embeddings.shape[-1]
        faiss_index = faiss.index_factory(dim, 'IVF100,PQ16', faiss.METRIC_L2)
        #faiss_index = faiss.index_factory(dim, 'Flat', faiss.METRIC_L2)
        faiss_index.train(all_embeddings)
        faiss_index.add(all_embeddings)
        faiss.write_index(faiss_index, self.save_dir + "/faiss_ivf.index")
        
        # build corpus databse
        db = Database(self.database_path)
        docs = db['docs']
        docs.insert_all(self.corpus, pk="id", batch_size=1000000, truncate=True)

        print("Finish!")



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description = "Creating index.")

#     # Basic parameters
#     parser.add_argument('--retrieval_method', type=str)
#     parser.add_argument('--corpus_path', type=str)
#     parser.add_argument('--save_dir', type=str)

#     # Parameters for building dense index
#     parser.add_argument('--device', type=str, default='cuda')
#     parser.add_argument('--gpu_num', type=int, default=1)
#     parser.add_argument('--doc_max_length', type=int, default=256)
#     parser.add_argument('--batch_size', type=int, default=512)
#     parser.add_argument('--use_fp16', type=bool, default=True)
    
#     args = parser.parse_args()

#     index_builder = Index_Builder(
#                         retrieval_method = args.retrieval_method,
#                         corpus_path = args.corpus_path,
#                         save_dir = args.save_dir,
#                         device = torch.device(args.device),
#                         gpu_num = args.gpu_num,
#                         doc_max_length = args.doc_max_length,
#                         batch_size = args.batch_size,
#                         retrieval_model_path = MODEL2PATH[args.retrieval_method],
#                         use_fp16 = args.use_fp16,
#                         pooling_method = MODEL2POOLING[args.retrieval_method],
#                     )
#     index_builder.build_index()