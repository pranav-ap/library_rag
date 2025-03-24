from langchain.schema import Document


def langchain_docs_to_string(docs: [Document]):
    return "\n\n".join(doc.page_content for doc in docs)
