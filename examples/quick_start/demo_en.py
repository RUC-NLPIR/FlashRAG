import streamlit as st
from flashrag.config import Config
from flashrag.utils import get_retriever, get_generator
from flashrag.prompt import PromptTemplate


config_dict = {
    "save_note": "demo",
    "model2path": {"e5": "intfloat/e5-base-v2", "llama3-8B-instruct": "meta-llama/Meta-Llama-3-8B-Instruct"},
    "retrieval_method": "e5",
    "generator_model": "llama3-8B-instruct",
    "corpus_path": "indexes/general_knowledge.jsonl",
    "index_path": "indexes/e5_Flat.index",
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
        "You are a friendly AI Assistant."
        "Respond to the input as a friendly AI assistant, generating human-like text, and follow the instructions in the input if applicable."
        "\nThe following are provided references. You can use them for answering question.\n\n{reference}"
    )
    system_prompt_no_rag = (
        "You are a friendly AI Assistant."
        "Respond to the input as a friendly AI assistant, generating human-like text, and follow the instructions in the input if applicable.\n"
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
