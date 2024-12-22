from utils import logger
from src import RAGSystem


if __name__ == "__main__":
    rag_system = RAGSystem()

    user_prompt = "Who is Sherlock Holmes?"
    logger.info('User Prompt')
    print(user_prompt)

    response = rag_system.query(user_prompt)

    logger.info('Response')
    print(response)
