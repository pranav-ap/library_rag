from config import config
from utils import logger

from smolagents import (
    Tool,
)

from sentence_transformers import CrossEncoder
from .DataStore import DataStore, bm25_tokenizer


class RetrieverTool(Tool):
    name = "Retriever Tool"
    description = "Retrieves snippets from locally available documents and books that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_store = DataStore()
        self.data_store.load()

        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def _get_chroma_results(self, user_prompt):
        results = self.data_store.chroma_collection.query(
            query_texts=user_prompt,
            n_results=config.task.n_results
        )

        # Results are already Sorted

        results = {
            "ids": results["ids"][0],
            "snippets": results["documents"][0],
            "distances": results["distances"][0]
        }

        return results

    def _get_bm25_results(self, user_prompt):
        tokenized_query = bm25_tokenizer(user_prompt)

        index = self.data_store.bm25_collection['index']
        ids = self.data_store.bm25_collection['ids']
        documents = self.data_store.bm25_collection['documents']

        scores = index.get_scores(tokenized_query)

        n_results = config.task.n_results

        # Sorting is important here. RRF assumes it.
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True)[:n_results]

        results = {
            "ids": [ids[i] for i in top_indices],
            "snippets": [documents[i] for i in top_indices],
            "scores": [scores[i] for i in top_indices]
        }

        return results

    @staticmethod
    def _rrf_score(keyword_rank: int, semantic_rank: int) -> float:
        k = 60
        return 1 / (k + keyword_rank) + 1 / (k + semantic_rank)

    def _combine_results(self, chroma_results, bm25_results):
        # Map Ids to Ranks
        chroma_ranks = {id_: index + 1 for index, id_ in enumerate(chroma_results['ids'])}
        bm25_ranks = {id_: index + 1 for index, id_ in enumerate(bm25_results['ids'])}

        # Combine rankings using Reciprocal Rank Fusion (RRF)
        combined_scores = {}
        for id_ in set(chroma_ranks.keys()).union(bm25_ranks.keys()):
            chroma_rank = chroma_ranks.get(id_, float('inf'))
            bm25_rank = bm25_ranks.get(id_, float('inf'))

            combined_scores[id_] = self._rrf_score(chroma_rank, bm25_rank)

        # Sort by combined scores (descending)
        sorted_ids = sorted(
            combined_scores.keys(),
            key=lambda id_: combined_scores[id_],
            reverse=True
        )

        # Retrieve corresponding snippets

        snippets = {
            id_: snippet for id_, snippet in zip(chroma_results['ids'], chroma_results['snippets'])
        }

        snippets.update({
            id_: snippet for id_, snippet in zip(bm25_results['ids'], bm25_results['snippets'])
        })

        combined_results = {
            "ids": sorted_ids,
            "snippets": [snippets[id_] for id_ in sorted_ids],
        }

        return combined_results

    def _re_rank(self, user_prompt, results) -> [str]:
        ids = results['ids']
        snippets = results['snippets']

        ranks = self.cross_encoder.rank(user_prompt, snippets)
        sorted_ranks = sorted(ranks, key=lambda x: x['score'], reverse=True)

        reranked_results = {
            'ids': [],
            'snippets': []
        }

        for rank in sorted_ranks:
            reranker_id = rank['corpus_id']

            _id = ids[reranker_id]
            snippet = snippets[reranker_id]

            reranked_results['ids'].append(_id)
            reranked_results['snippets'].append(snippet)

        return reranked_results

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        chroma_results = self._get_chroma_results(query)
        bm25_results = self._get_bm25_results(query)
        retrieval_results = self._combine_results(chroma_results, bm25_results)
        retrieval_results = self._re_rank(query, retrieval_results)

        snippets = retrieval_results['snippets']

        memory = "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n" + snippet
                for i, snippet in enumerate(snippets)
            ]
        )

        return memory
