import streamlit as st
from flashrag.config import Config
from flashrag.utils import get_retriever, get_generator
from flashrag.prompt import PromptTemplate

config_dict = {
    "save_note": "demo",
    "generator_model": "qwen-14B",
    "retrieval_method": "bge-zh",
    "model2path": {"bge-zh": "BAAI/bge-large-zh-v1.5", "qwen-14B": "Qwen/Qwen1.5-14B-Chat"},
    "corpus_path": "/data00/jiajie_jin/rd_corpus.jsonl",
    "index_path": "/data00/jiajie_jin/flashrag_indexes/rd_corpus/bge_Flat.index",
}


@st.cache_resource
def load_retriever(_config):
    return get_retriever(_config)


@st.cache_resource
def load_generator(_config):
    return get_generator(_config)

if __name__ == '__main__':
    custom_theme = {
        "primaryColor": "#ff6347",
        "backgroundColor": "#f0f0f0",
        "secondaryBackgroundColor": "#d3d3d3",
        "textColor": "#121212",
        "font": "sans serif",
    }
    st.set_page_config(page_title="FlashRAG Demo", page_icon="⚡")


    st.sidebar.title("Configuration")
    temperature = st.sidebar.slider("Temperature:", 0.01, 1.0, 0.5)
    topk = st.sidebar.slider("Number of retrieved documents:", 1, 10, 5)
    max_new_tokens = st.sidebar.slider("Max generation tokens:", 1, 2048, 256)


    st.title("⚡FlashRAG Demo")
    st.write("This demo retrieves documents and generates responses based on user input.")


    query = st.text_area("Enter your prompt:")

    config = Config("my_config.yaml", config_dict=config_dict)
    generator = load_generator(config)
    retriever = load_retriever(config)

    system_prompt_rag = (
        "你是一个友好的人工智能助手。"
        "请对用户的输出做出高质量的响应，生成类似于人类的内容，并尽量遵循输入中的指令。"
        "\n下面是一些可供参考的文档，你可以使用它们来回答问题。\n\n{reference}"
    )
    system_prompt_no_rag = (
        "你是一个友好的人工智能助手。" "请对用户的输出做出高质量的响应，生成类似于人类的内容，并尽量遵循输入中的指令。\n"
    )
    base_user_prompt = "{question}"

    prompt_template_rag = PromptTemplate(config, system_prompt=system_prompt_rag, user_prompt=base_user_prompt)
    prompt_template_no_rag = PromptTemplate(config, system_prompt=system_prompt_no_rag, user_prompt=base_user_prompt)


    if st.button("Generate Responses"):
        with st.spinner("Retrieving and Generating..."):
            retrieved_docs = retriever.search(query, num=topk)

            st.subheader("References", divider="gray")
            for i, doc in enumerate(retrieved_docs):
                doc_title = doc.get("title", "No Title")
                doc_text = "\n".join(doc["contents"].split("\n")[1:])
                expander = st.expander(f"**[{i+1}]: {doc_title}**", expanded=False)
                with expander:
                    st.markdown(doc_text, unsafe_allow_html=True)

            st.subheader("Generated Responses:", divider="gray")

            input_prompt_with_rag = prompt_template_rag.get_string(question=query, retrieval_result=retrieved_docs)
            response_with_rag = generator.generate(
                input_prompt_with_rag, temperature=temperature, max_new_tokens=max_new_tokens
            )[0]
            st.subheader("Response with RAG:")
            st.write(response_with_rag)
            input_prompt_without_rag = prompt_template_no_rag.get_string(question=query)
            response_without_rag = generator.generate(
                input_prompt_without_rag, temperature=temperature, max_new_tokens=max_new_tokens
            )[0]
            st.subheader("Response without RAG:")
            st.markdown(response_without_rag)
