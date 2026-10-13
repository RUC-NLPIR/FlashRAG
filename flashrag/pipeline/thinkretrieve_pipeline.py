"""FlashRAG integration sketch for reasoning-time worked-example retrieval.

FlashRAG's standard retriever returns documents before generation. This adapter
keeps a separate ThinkRetrieve example bank and delegates the generation loop
to the library, so the retrieved worked solution is injected after an interim
reasoning probe. It intentionally imports FlashRAG lazily: the repository's
core installation does not need FlashRAG.

The ``dataset`` object follows FlashRAG's Dataset convention: it must expose a
``question`` sequence and accepts ``update_output(key, values)``. A compatible
generator can be supplied by passing any OpenAI-compatible endpoint through
ThinkRetrieve's existing backend.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from thinkretrieve import (
    FaissRetriever,
    OpenAICompatBackend,
    ThinkRetrieve,
    ThinkRetrieveConfig,
)


class ThinkRetrievePipeline:
    """Run ThinkRetrieve over a FlashRAG-style dataset."""

    def __init__(
        self,
        bank_path: str,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "EMPTY",
        config: Optional[ThinkRetrieveConfig] = None,
        backend: Any = None,
    ) -> None:
        self.retriever = FaissRetriever.load(bank_path)
        self.backend = backend or OpenAICompatBackend(
            model=model, base_url=base_url, api_key=api_key
        )
        self.config = config or ThinkRetrieveConfig()

    def run(self, dataset: Any, do_eval: bool = True, pred_process_fun: Optional[Callable] = None) -> Any:
        del do_eval
        answers = []
        traces = []
        retrieval_counts = []
        for question in dataset.question:
            result = ThinkRetrieve(
                self.backend, self.retriever, self.config
            ).run(question)
            answer = result.answer.strip()
            if pred_process_fun is not None:
                answer = pred_process_fun(answer)
            answers.append(answer)
            traces.append(result.think_trace)
            retrieval_counts.append(len(result.retrievals))
        dataset.update_output("pred", answers)
        dataset.update_output("thinkretrieve_trace", traces)
        dataset.update_output("thinkretrieve_retrievals", retrieval_counts)
        return dataset
