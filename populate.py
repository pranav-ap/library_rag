from src import DataStore


if __name__ == "__main__":
    store = DataStore(reset=True)
    store.populate()
