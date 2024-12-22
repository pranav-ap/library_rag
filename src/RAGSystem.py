import ollama
from sentence_transformers import CrossEncoder

from config import config
from utils import logger
from .DataStore import DataStore, bm25_tokenizer
from .prompt import PromptTemplate


class RAGSystem:
    def __init__(self):
        self.data_store = DataStore()
        self.data_store.load()

        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    @staticmethod
    def _rewrite_query(user_prompt):
        preamble = """
        Your are my AI assistant. 
        Reformulate the user prompt to improve retrieval in a RAG system. 
        """

        preferences = [
            'Do not change the meaning of the user prompt.',
            'Retain important keywords and phrases.',
        ]

        prompt_template = PromptTemplate(
            preamble=preamble,
            preferences=preferences,
        )

        prompt: str = prompt_template.prepare_rewrite_prompt(
            user_prompt=user_prompt
        )

        response = ollama.generate(
            model=config.task.llm,
            prompt=prompt,
            options={
                "max_tokens": config.task.max_tokens,
                "temperature": config.task.temperature,
            }
        )

        response = response['response']

        if config.task.debug_mode:
            logger.info('Rewritten User Prompt')
            print(response)

        return response

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

        logger.info(f'Chroma Result Count : {len(results['ids'])}')

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

        logger.info(f'BM25 Result Count : {len(results['ids'])}')

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

        logger.info(f'Hybrid Result Count : {len(combined_results["ids"])}')

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

    @staticmethod
    def _final_query(user_prompt, retrieval_results):
        preferences = [
            'Answer in simple words within 100 words. Omit supplementary details.',
            'Only use context information. Nothing else.',
        ]

        preamble = """
        Your are my AI assistant. You answer the user question according to my preferences using the given context.
        My Preferences are listed below:
        """

        snippets = retrieval_results['snippets']

        prompt_template = PromptTemplate(
            preamble=preamble,
            preferences=preferences,
        )

        prompt: str = prompt_template.prepare_simple_prompt(
            snippets=snippets,
            user_prompt=user_prompt
        )

        if config.task.debug_mode:
            print(prompt)
            logger.info(f'Prompt Length : {len(prompt)}')

        response = ollama.generate(
            model=config.task.llm,
            prompt=prompt,
            options={
                "max_tokens": config.task.max_tokens,
                "temperature": config.task.temperature,
            }
        )

        response = response['response']

        return response

    def query(self, user_prompt):
        if config.task.rewrite_user_prompt:
            user_prompt = self._rewrite_query(user_prompt)

        chroma_results = self._get_chroma_results(user_prompt)
        bm25_results = self._get_bm25_results(user_prompt)

        retrieval_results = self._combine_results(chroma_results, bm25_results)

        if config.task.rerank:
            logger.info('Re-ranking Snippets')
            retrieval_results = self._re_rank(
                user_prompt,
                retrieval_results,
            )

        response = self._final_query(user_prompt, retrieval_results)

        return response
