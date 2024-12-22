from typing import List


class PromptTemplate:
    @classmethod
    def prepare_context(cls, snippets: List[str], empty_space: int) -> str:
        context = ''

        for i, text in enumerate(snippets):
            context += f"Context {i+1}:\n{text}\n"

        limit = min(empty_space, len(context))
        context = context[:limit]

        return context
