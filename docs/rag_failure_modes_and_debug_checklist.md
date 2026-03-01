# RAG failure modes and debug checklist

This page gives a small, optional debugging checklist for teams that use FlashRAG to reproduce RAG baselines or build new pipelines.

The goal is practical: if your experiments run but the results look strange, you can skim this page and quickly decide **which part of the pipeline to inspect first**, instead of guessing blindly.

This page is documentation only. It does not change any code or APIs in FlashRAG.

We use the following tags:

- `[IN]` input and retrieval  
- `[RE]` reasoning and planning  
- `[ST]` state and context  
- `[OP]` infra and deployment  
- `{OBS}` observability and evaluation  
- `{SEC}` security  
- `{LOC}` language and OCR  

---

## 1. Quick map: sixteen failure modes

This table is a generic map of reproducible failure modes that show up across RAG and agent pipelines. You can treat it as a shared vocabulary when discussing incidents.

| # | problem domain (with layer / tags) | what breaks |
| --- | --- | --- |
| 1 | [IN] hallucination & chunk drift {OBS} | retrieval returns wrong or irrelevant content |
| 2 | [RE] interpretation collapse | chunk is right, logic is wrong |
| 3 | [RE] long reasoning chains {OBS} | drifts across multi-step tasks |
| 4 | [RE] bluffing / overconfidence | confident but unfounded answers |
| 5 | [IN] semantic ≠ embedding {OBS} | cosine match ≠ true meaning |
| 6 | [RE] logic collapse & recovery {OBS} | dead-ends, needs controlled reset |
| 7 | [ST] memory breaks across sessions | lost threads, no continuity |
| 8 | [IN] debugging is a black box {OBS} | no visibility into failure path |
| 9 | [ST] entropy collapse | attention melts, incoherent output |
| 10 | [RE] creative freeze | flat, literal outputs |
| 11 | [RE] symbolic collapse | abstract or logical prompts break |
| 12 | [RE] philosophical recursion | self-reference loops, paradox traps |
| 13 | [ST] multi-agent chaos {OBS} | agents overwrite or misalign logic |
| 14 | [OP] bootstrap ordering | services fire before dependencies are ready |
| 15 | [OP] deployment deadlock | circular waits inside infra |
| 16 | [OP] pre-deploy collapse {OBS} | version skew or missing secret on first call |

You do not have to use all sixteen at once. In practice, most incidents fall into a small subset of rows.

---

## 2. Using this checklist with FlashRAG

FlashRAG provides components for data loading, indexing, retrieval, generation, and evaluation. When something feels off, a simple workflow is:

1. **Collect a few concrete failures**

   For one configuration (dataset + method), pick 10–20 examples where:
   - the score is clearly worse than expected, or  
   - the output looks wrong to a human, even though the pipeline runs.

2. **Map each failure to a row**

   For each example, ask “which row in the table does this most resemble?”.  
   A few typical mappings in FlashRAG:

   - Retrieval looks unrelated, or answer spans are missing → often No.1 or No.5  
   - Retrieved chunks look fine, but answers drift or mix concepts → often No.2, No.3 or No.6  
   - Logs do not make it clear *why* a document was retrieved → often No.8  
   - First production-like runs fail because an index or model was not fully ready → often No.14 or No.16  

   The match does not have to be perfect. The point is to label the *pattern*, not a specific bug.

3. **Group by failure mode, then inspect the right layer**

   Once you have labels, group incidents by their row number:

   - If many failures are No.1 / No.5  
     → focus on corpus, chunking, embedding model, and retrieval configuration.  
   - If many failures are No.2 / No.3 / No.6  
     → focus on prompts, reasoning depth, and how the generator consumes retrieved passages.  
   - If many failures are No.8  
     → focus on logging, attribution, and tools that show “query → retrieved docs → answer” end to end.  
   - If many failures are No.14 / No.16  
     → focus on run order, index-building steps, and environment / version consistency.

4. **Make a small change, then re-check**

   After adjusting one part of the pipeline (for example, retriever settings or prompt templates):

   - re-run the same configuration on a small subset  
   - check whether incidents moved from one row to another, or actually disappeared  
   - keep a short log of “before / after” examples to avoid regressions

This process does not require any new tools. It only adds a structured way to describe what is going wrong.

---

## 3. Bug report checklist for FlashRAG

When opening a GitHub issue or asking for help, including the items below makes it much easier to reason about failure modes:

- **Dataset**: name and split (e.g., `hotpot_qa`, validation or test subset)
- **Method / config**: method name or config file path
- **Retriever / index**: retriever type, index type, where and how the index was built
- **Generator**: model name and decoding settings
- **Evaluation**: metric, evaluation script, and any normalization options
- **Seeds**: random seeds used (if any)
- **Logs**: a short log snippet showing:
  - the question  
  - retrieved document IDs or short excerpts  
  - the final answer and gold answer  

Providing this information turns “it looks wrong” into a concrete, reproducible debugging task.

---

## Credits

This checklist is adapted from the open-source [WFGY ProblemMap](https://github.com/onestardao/WFGY/blob/main/ProblemMap/README.md), a 16-mode taxonomy of reproducible failure modes for RAG pipelines and agents (MIT-licensed). The intention here is not to change FlashRAG itself, but to offer a lightweight debugging lens for users when experiments behave in unexpected ways.
