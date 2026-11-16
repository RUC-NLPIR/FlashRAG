import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from flashrag.retriever.retriever import BaseTextRetriever
from flashrag.version import __version__

_PARALLEL_MCP_URL = "https://mcp.parallel.ai/mcp"


async def _normalize_media_type(response):
    # The SSE reader expects a lowercase media type.
    media_type, separator, parameters = response.headers.get("Content-Type", "").partition(";")
    if media_type != media_type.lower():
        response.headers["Content-Type"] = media_type.lower() + separator + parameters


class ParallelRetriever(BaseTextRetriever):
    """Retrieve text excerpts from the web using the anonymous Parallel MCP server."""

    def _search(self, query, num=None, return_score=False):
        documents, scores = self._batch_search([query], num, return_score=True)
        return (documents[0], scores[0]) if return_score else documents[0]

    def _batch_search(self, query, num=None, return_score=False):
        num = self.topk if num is None else num
        queries = [query] if isinstance(query, str) else query
        queries = [item.strip() for item in queries]

        if num <= 0 or not any(queries):
            documents = [[] for _ in queries]
        else:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                documents = asyncio.run(self._search_queries(queries, num))
            else:
                # The public API is synchronous, including when called from a notebook.
                with ThreadPoolExecutor(max_workers=1) as executor:
                    documents = executor.submit(asyncio.run, self._search_queries(queries, num)).result()

        if return_score:
            # The server returns results in relevance order without numeric scores.
            scores = [[1.0 / rank for rank in range(1, len(row) + 1)] for row in documents]
            return documents, scores
        return documents

    async def _search_queries(self, queries, num):
        try:
            import httpx
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
        except ModuleNotFoundError as exc:
            if exc.name not in {"httpx", "mcp"}:
                raise
            raise ImportError(
                "Parallel retrieval requires Python 3.10+ and the parallel extra. "
                "Run pip install -e '.[parallel]' from the FlashRAG checkout."
            ) from exc

        class SessionClient(httpx.AsyncClient):
            async def send(self, request, *, stream=False, **kwargs):
                if request.method == "DELETE":
                    # Bound the whole cleanup, including a slowly streamed response body.
                    return await asyncio.wait_for(super().send(request, stream=False, **kwargs), timeout=1.0)
                return await super().send(request, stream=stream, **kwargs)

        async with (
            SessionClient(
                headers={"User-Agent": f"FlashRAG/{__version__}"},
                timeout=30.0,
                follow_redirects=False,
                max_redirects=0,  # The MCP SDK also handles same-origin redirects.
                event_hooks={"response": [_normalize_media_type]},
            ) as client,
            streamable_http_client(_PARALLEL_MCP_URL, http_client=client) as (read, write, _),
            ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30)) as session,
        ):
            await session.initialize()
            rows = []
            for query in queries:
                rows.append(await self._search_query(session, query, num) if query else [])
            return rows

    @staticmethod
    async def _search_query(session, query, num):
        result = await session.call_tool(
            "web_search",
            {
                "objective": f"Find source passages that answer this query: {query}",
                "search_queries": [query],
            },
        )
        if result.isError:
            detail = next((block.text for block in result.content if block.type == "text"), "Unknown tool error")
            raise RuntimeError(f"Parallel web search failed: {detail}")
        if result.structuredContent is None or "results" not in result.structuredContent:
            raise RuntimeError("Parallel web search returned no structured results")

        documents = []
        for item in result.structuredContent["results"]:
            url = item["url"]
            text = "\n\n".join(excerpt.strip() for excerpt in item["excerpts"] if excerpt.strip())
            if not url.startswith(("https://", "http://")) or not text:
                continue
            title = " ".join((item.get("title") or "").split()) or url
            documents.append({"id": url, "title": title, "text": text, "url": url, "contents": f"{title}\n{text}"})
            if len(documents) == num:
                break
        return documents
