# RAG

## Components



## Evaluation

First set a local model as evaluation model.

```commandline
deepeval set-local-model --model-name=gemma2:9b --base-url="http://localhost:11434/v1/"
```

Then run evaluate.

```commandline
python .\evaluate.py
```


