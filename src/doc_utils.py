from langchain.schema import Document


def langchain_docs_to_single_string(docs: [Document]):
    return "\n\n".join(doc.page_content for doc in docs)


def replace_t_with_space(docs: [Document]):
    for doc in docs:
        doc.page_content = doc.page_content.replace('\t', ' ')

    return docs
