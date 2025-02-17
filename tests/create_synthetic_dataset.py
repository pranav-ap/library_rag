from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset


synthesizer = Synthesizer()

goldens = synthesizer.generate_goldens_from_docs(
    document_paths=['data/The Adventures of Sherlock Holmes.txt']
)

dataset = EvaluationDataset(goldens=goldens)

dataframe = synthesizer.to_pandas()
print(dataframe)

synthesizer.save_as(
    file_type='json',
    directory="./synthetic_data"
)
