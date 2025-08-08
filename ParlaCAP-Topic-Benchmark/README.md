# CAP Topic Classification in Parliamentary Proceedings Benchmark

A benchmark for evaluating performance of various classification models on CAP (Comparative Agendas Project) topics for parliamentary proceedings.

The multilingual fine-tuned ParlaCAP classifier is freely available at the HuggingFace repository: https://huggingface.co/classla/ParlaCAP-Topic-Classifier

The code for all evaluated models is available in the [systems](systems) directory.

## Benchmark scores



![](evaluation-for-the-paper/parlacap-results-heatmap.png)

------------------------------------------



## Contributing to the benchmark

Should you wish to contribute an entry, feel free to submit a folder in the [systems](systems) directory with or without the code used (see the submission examples in the directory).

The results JSON file name should start with `submission-` and the content should be structured like this:

```python
{
    "system": "Pick a name for your system",
    "predictions": [
        {   "train": "what you trained on", # e.g. "ParlaCAP-train"
            "test": "what you evaluated on", # should be "ParlaCAP-HR-test" or "ParlaCAP-EN-test"
            "predictions": [....] # The length of predictions should match the length of test data
        },
    ],
    # Additional information, e.g. fine-tuning params:
    "model": "EMBEDDIA/crosloengual-bert",
    "lr": "4e-5",
    "epoch": "15"
}
```

All submission JSON files should be saved in a `submissions` directory inside the directory for your system. They will be evaluated against the datasets in the `data/datasets` directory.

It is highly encouraged that you also provide additional information about your system in a README file, and that you provide the code used for the classification with the system.

## Evaluation

Micro and Macro F1 scores will be used to evaluate and compare systems.

The submissions are evaluated using the following code with the path to the submissions directory (e.g., ``systems/dummy-classifier/submissions``) as the argument. The log file is to be saved in the relevant system directory:
```python eval.py "submission-path" > systems/dummy-classifier/evaluation.log```

The code produces:
- a JSON file with the results of all tested models: `results/results.json`
- a table with the results, e.g. `results/results-ParlaCAP-HR-test.md`
