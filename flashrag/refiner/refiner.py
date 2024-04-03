from typing import List, Dict
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from tqdm import tqdm

class BasicRefiner:
    r"""Base object of Refiner method"""

    def __init__(self, config):
        self.config = config
        self.model_path = config['refiner_model_path']
        self.device = config['device']
    
    def run(self, item) -> str:
        r"""Get refining result.

        Args:
            item: dataset item, contains question, retrieval result...

        Returns:
            str: refining result of this item
        """
        pass

    def batch_run(self, dataset, batch_size = None) -> List[str]:
        return [self.run(item) for item in dataset]



class AbstractiveRecompRefiner(BasicRefiner):
    """Implementation for Abstractive RECOMP compressor: 
        RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation.
    """

    def __init__(self, config):
        super().__init__(config)
        
        self.max_input_length = 1024
        self.max_output_length = 512
        
        # load model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        self.model.cuda()
        self.model.eval()


    
    def batch_run(self, dataset, batch_size = 2):
        # input processing in recomp training format
        format_inputs = ['Question: {question}\n Document: {document}\n Summary: '.format(
            question = item.question,
            document = "\n".join(item.retrieval_result)
        ) for item in dataset]

        results = []
        for idx in tqdm(range(0, len(format_inputs), batch_size), desc='Refining process: '):
            batch_inputs = format_inputs[idx:idx+batch_size]
            batch_inputs = self.tokenizer(batch_inputs,
                                    return_tensors='pt',
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_input_length
                                ).to(self.device)


            batch_outputs = self.model.generate(
                **batch_inputs,
                max_length=self.max_output_length
            )

            batch_outputs = self.tokenizer.batch_decode(
                batch_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            results.extend(batch_outputs)    
        
        return results
