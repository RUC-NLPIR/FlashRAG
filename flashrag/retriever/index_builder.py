import os
import faiss
import json
import warnings
import numpy as np
from typing import cast
import shutil
import subprocess
import argparse
import torch
from tqdm import tqdm
from flashrag.retriever.utils import load_model, load_corpus, pooling

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
            faiss_type=None,
            embedding_path=None,
            save_embedding=False,
            faiss_gpu=False,
            use_sentence_transformer=False
        ):

        self.retrieval_method = retrieval_method.lower()
        self.model_path = model_path
        self.corpus_path = corpus_path
        self.save_dir = save_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.pooling_method = pooling_method
        self.faiss_type = faiss_type if faiss_type is not None else 'Flat'
        self.embedding_path = embedding_path
        self.save_embedding = save_embedding
        self.faiss_gpu = faiss_gpu
        self.use_sentence_transformer = use_sentence_transformer

        self.gpu_num = torch.cuda.device_count()
        # prepare save dir
        print(self.save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            if not self._check_dir(self.save_dir):
                warnings.warn("Some files already exists in save dir and may be overwritten.", UserWarning)

        self.index_save_path = os.path.join(self.save_dir, f"{self.retrieval_method}_{self.faiss_type}.index")

        self.embedding_save_path = os.path.join(self.save_dir, f"emb_{self.retrieval_method}.memmap")

        self.corpus = load_corpus(self.corpus_path)

        print("Finish loading...")
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
        shutil.copyfile(self.corpus_path, temp_file_path)

        print("Start building bm25 index...")
        pyserini_args = ["--collection", "JsonCollection",
                         "--input", temp_dir,
                         "--index", self.save_dir,
                         "--generator", "DefaultLuceneDocumentGenerator",
                         "--threads", "1"]

        subprocess.run(["python", "-m", "pyserini.index.lucene"] + pyserini_args)

        shutil.rmtree(temp_dir)

        print("Finish!")

    def _load_embedding(self, embedding_path, corpus_size, hidden_size):
        all_embeddings = np.memmap(
                embedding_path,
                mode="r",
                dtype=np.float32
            ).reshape(corpus_size, hidden_size)
        return all_embeddings

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

    def st_encode_all(self):
        if self.gpu_num > 1:
            print("Use multi gpu!")
            self.batch_size = self.batch_size * self.gpu_num

        sentence_list = [item['contents'] for item in self.corpus]
        if self.retrieval_method == "e5":
            sentence_list = [f"passage: {doc}" for doc in sentence_list]
        all_embeddings = self.encode(
            sentence_list, 
            batch_size = self.batch_size
        )

        return all_embeddings

    def encode_all(self):
        if self.gpu_num > 1:
            print("Use multi gpu!")
            self.encoder = torch.nn.DataParallel(self.encoder)
            self.batch_size = self.batch_size * self.gpu_num

        all_embeddings = []

        for start_idx in tqdm(range(0, len(self.corpus), self.batch_size), desc='Inference Embeddings:'):
            batch_data = self.corpus[start_idx:start_idx+self.batch_size]['contents']

            if self.retrieval_method == "e5":
                batch_data = [f"passage: {doc}" for doc in batch_data]

            inputs = self.tokenizer(
                        batch_data,
                        padding=True,
                        truncation=True,
                        return_tensors='pt',
                        max_length=self.max_length,
            ).to('cuda')

            inputs = {k: v.cuda() for k, v in inputs.items()}

            #TODO: support encoder-only T5 model
            if "T5" in type(self.encoder).__name__ or (self.gpu_num > 1 and "T5" in type(self.encoder.module).__name__):
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

        return all_embeddings

    @torch.no_grad()
    def build_dense_index(self):
        """Obtain the representation of documents based on the embedding model(BERT-based) and 
        construct a faiss index.
        """

        if os.path.exists(self.index_save_path):
            print("The index file already exists and will be overwritten.")

        if self.use_sentence_transformer:
            from flashrag.retriever.encoder import STEncoder
            self.encoder = STEncoder(
                model_name = self.retrieval_method,
                model_path = self.model_path,
                max_length = self.max_length,
                use_fp16 = self.use_fp16
            )
            hidden_size = self.encoder.model.get_sentence_embedding_dimension()
        else:
            self.encoder, self.tokenizer = load_model(model_path = self.model_path,
                                                    use_fp16 = self.use_fp16)
            hidden_size = self.encoder.config.hidden_size

        if self.embedding_path is not None:
            corpus_size = len(self.corpus)
            all_embeddings = self._load_embedding(self.embedding_path, corpus_size, hidden_size)
        else:
            all_embeddings = self.st_encode_all() if self.use_sentence_transformer else self.encode_all()
            if self.save_embedding:
                self._save_embedding(all_embeddings)
            del self.corpus

        # build index
        print("Creating index")
        dim = all_embeddings.shape[-1]
        faiss_index = faiss.index_factory(dim, self.faiss_type, faiss.METRIC_INNER_PRODUCT)

        if self.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)
            if not faiss_index.is_trained:
                faiss_index.train(all_embeddings)
            faiss_index.add(all_embeddings)
            faiss_index = faiss.index_gpu_to_cpu(faiss_index)
        else:
            if not faiss_index.is_trained:
                faiss_index.train(all_embeddings)
            faiss_index.add(all_embeddings)

        faiss.write_index(faiss_index, self.index_save_path)
        print("Finish!")



MODEL2POOLING = {
    "e5": "mean",
    "bge": "cls",
    "contriever": "mean",
    'jina': 'mean'
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
    parser.add_argument('--embedding_path', default=None, type=str)
    parser.add_argument('--save_embedding', action='store_true', default=False)
    parser.add_argument('--faiss_gpu', default=False, action='store_true')
    parser.add_argument('--sentence_transformer', action='store_true', default=False)

    args = parser.parse_args()

    if args.pooling_method is None:
        pooling_method = 'mean'
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
                        faiss_type = args.faiss_type,
                        embedding_path = args.embedding_path,
                        save_embedding = args.save_embedding,
                        faiss_gpu = args.faiss_gpu,
                        use_sentence_transformer = args.sentence_transformer
                    )
    index_builder.build_index()


if __name__ == "__main__":
    main()