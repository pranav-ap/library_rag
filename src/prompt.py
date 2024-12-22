from typing import List
from config import config
from utils import logger


class PromptTemplate:
    def __init__(self, preamble: str, preferences: [str]):
        self.preamble = preamble
        self.preferences = '\n - '.join(preferences) if len(preferences) > 0 else ''

    @staticmethod
    def prepare_context(snippets: List[str], empty_space: int) -> str:
        context = ''

        for i, text in enumerate(snippets):
            context += f"Context {i+1}:\n{text}\n"

        if empty_space < len(context):
            logger.warning(f"Context is too long. Truncating to {empty_space} characters.")

        limit = min(empty_space, len(context))
        context = context[:limit]

        return context

    def prepare_simple_prompt(
            self,
            snippets: [str],
            user_prompt: str
    ) -> str:
        empty_space = config.llm[config.task.llm].context_window_size
        empty_space = empty_space - len(self.preamble) - len(self.preferences) - len(user_prompt) - 50

        context = self.prepare_context(snippets, empty_space)

        prompt = f"""
        <|system|>
        {self.preamble}
        {self.preferences}<|end|>
        <|user|>
        Context:
        {context}
        Question:
        {user_prompt}<|end|>
        <|assistant|>"
        """

        return prompt

    def prepare_rewrite_prompt(
            self,
            user_prompt: str
    ) -> str:
        prompt = f"""
        <|system|>
        {self.preamble}
        {self.preferences}<|end|>
        <|user|>
        Original User Prompt:
        {user_prompt}<|end|>
        <|assistant|>"
        """

        return prompt

