from flashrag.config import Config
from flashrag.dataset import Dataset, Item


config_dict = {
    "save_note": 'test',
    'refiner_name': 'longllmlingua',
    'refiner_model_path': 'model/llama-2-7b-hf',
    'refiner_input_prompt_flag': True, # 直接对prompt进行压缩
    'llmlinuga_config': {
        'use_llmlingua2': False,
        "rate": 0.55,
        "condition_in_question": "after_condition",
        "reorder_context": "sort",
        "dynamic_context_compression_ratio": 0.3,
        "condition_compare": True,
        "context_budget": "+100",
        "rank_method": "longllmlingua",
    }
}

config = Config('my_config.yaml', config_dict)

from flashrag.refiner import LLMLinguaRefiner

refiner = LLMLinguaRefiner(config)

prompt = "Answer the question based on the given document. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n\nDoc 1(Title: \"Polish-Russian War (film)\") Polish-Russian War (film) Polish-Russian War (Wojna polsko-ruska) is a 2009 Polish film directed by Xawery \u017bu\u0142awski based on the novel Polish-Russian War under the white-red flag by Dorota Mas\u0142owska. The film's events take place over several days and they are set in the present time in a large Polish city. The main character is a bandit, a Polish dres (a Polish chav) called \"\"Strong\"\" (Borys Szyc), who does not work or study, and who frequently gets into conflict with the law and is in love with Magda (Roma G\u0105siorowska). The relationship is not going well. \n\nQuestion: Who is the mother of the director of film Polish-Russian War (Film)?"
dataset = Dataset(
    config = config, data = [Item({"prompt": prompt, "retrieval_result": ""})]
)
output = refiner.batch_run(dataset)[0]
print(output)