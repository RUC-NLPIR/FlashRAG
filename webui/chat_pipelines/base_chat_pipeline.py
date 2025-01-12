class BaseChatPipeline:
    def __init__(self, config):
        self.config = config
        self.save_retrieval_cache = config['save_retrieval_cache']
    
    def format_reference_chat(self, retrieval_result):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"    <strong><u>Doc {idx+1}(Title: {title})</u></strong> {text}\n"

        return format_reference

    def display_retrieval_result(self, retrieval_results):
        yield "<strong>Retrieval Result:</strong>\n" + self.format_reference_chat(retrieval_results)
    
    def display_middle_result(self, middle_result, display_message):
        if isinstance(middle_result, list):
            middle_result = middle_result[0]
        yield f"<strong>{display_message}:</strong>\n" + middle_result
    

