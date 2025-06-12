from src import DocumentIndexer


def main():
    paths = [
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://www.anthropic.com/engineering/building-effective-agents",
        "D:/Library/The British Empire_ Sunrise to Sunset.pdf",
        "D:/code/library_rag/data/The King.pdf",
    ]

    print(f'Indexing {len(paths)} documents...')

    indexer = DocumentIndexer()
    indexer.index_documents(paths)

    print('Done!')


if __name__ == "__main__":
    main()
