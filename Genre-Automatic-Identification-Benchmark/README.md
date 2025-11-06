# Genre Identification Benchmark

A benchmark for evaluating robustness of automatic genre identification models to test their usability for the automatic enrichment of large text collections with genre information.

For a more comprehensive benchmark that includes more languages, see the [AGILE Automatic Genre Identification Benchmark](https://github.com/TajaKuzman/AGILE-Automatic-Genre-Identification-Benchmark).

The benchmark is based on two manually-annotated test datasets:
- English EN-GINCO test dataset
- X-GINCO test dataset (the Croatian, Macedonian, and Slovenian part)

Both datasets are available upon request - please write to taja.kuzman@ijs.si to get access to private GitHub repositories where they are deposited.

The datasets are not publicly available to prevent exposure to LLM models which would make any further evaluation of the capabilities of generative large language models on this test set unreliable. If you wish to contribute to the benchmark, the test datasets will be shared with you upon request.

The X-GENRE classifier, which is state-of-the-art for this task, is freely available at the HuggingFace repository: https://huggingface.co/classla/xlm-roberta-base-multilingual-text-genre-classifier

The test datasets follow the same structure and genre schema as the [X-GENRE dataset](https://huggingface.co/datasets/TajaKuzman/X-GENRE-text-genre-dataset) on which the X-GENRE classifier and other neural and non-neural classifiers were trained on.

The code for all evaluated models is available in the [systems](systems) directory.

## Benchmark scores

Benchmark scores were calculated only once per system. Fine-tuning hyperparameters are listed in the json submission files, where applicable.

All models that were not used in a zero-shot scenario were trained on the train split of the [X-GENRE dataset](https://huggingface.co/datasets/TajaKuzman/X-GENRE-text-genre-dataset) which comprises manually-annotated instances in Slovenian and English language. As the EN-GINCO test dataset comprises English instances, the performance of the trained models is observed in a cross-dataset scenario.

The performance on EN-GINCO is generally lower than on X-GINCO datasets, because X-GINCO datasets contain only concrete labels, while EN-GINCO also has instances annotated as "Other".

![](evaluation-for-the-paper/genre-results-heatmap.png)

![](evaluation-for-the-paper/gpt_comparison.png)

| Model                      |   Slovenian (macro-F1) |   Slovenian (micro-F1) |   Croatian (macro-F1) |   Croatian (micro-F1) |   Macedonian (macro-F1) |   Macedonian (micro-F1) |   English (macro-F1) |   English (micro-F1) |
|:---------------------------|-----------------------:|-----------------------:|----------------------:|----------------------:|------------------------:|------------------------:|---------------------:|---------------------:|
| Fine-Tuned BERT-Like Model |                   0.94 |                   0.94 |                  0.89 |                  0.9  |                    0.91 |                    0.91 |                 0.75 |                 0.71 |
| GPT-4o                     |                   0.8  |                   0.8  |                  0.77 |                  0.78 |                    0.78 |                    0.8  |                 0.75 |                 0.77 |
| GPT-5                      |                   0.79 |                   0.79 |                  0.72 |                  0.74 |                    0.72 |                    0.71 |                 0.84 |                 0.83 |
| LLaMA 3.3                  |                   0.76 |                   0.76 |                  0.77 |                  0.78 |                    0.76 |                    0.76 |                 0.7  |                 0.77 |
| Mistral Medium 3.1         |                   0.75 |                   0.78 |                  0.71 |                  0.74 |                    0.74 |                    0.76 |                 0.72 |                 0.78 |
| Gemini 2.5 Flash           |                   0.72 |                   0.72 |                  0.77 |                  0.79 |                    0.67 |                    0.68 |                 0.79 |                 0.81 |
| Gemma 3                    |                   0.67 |                   0.66 |                  0.72 |                  0.74 |                    0.69 |                    0.7  |                 0.71 |                 0.74 |
| Qwen 3                     |                   0.66 |                   0.66 |                  0.65 |                  0.66 |                    0.61 |                    0.6  |                 0.6  |                 0.66 |
| GPT-3.5-Turbo              |                   0.59 |                   0.59 |                  0.59 |                  0.59 |                    0.51 |                    0.51 |                 0.57 |                 0.65 |
| Dummy (Frequent)           |                   0.03 |                   0.12 |                  0.03 |                  0.11 |                    0.03 |                    0.15 |                 0.04 |                 0.18 |


------------------------------------------

## Contributing to the benchmark

Should you wish to contribute an entry, feel free to submit a folder in the [systems](systems) directory with or without the code used (see the submission examples in the directory).

The results JSON file name should start with `submission-` and the content should be structured like this:

```python
{
    "system": "Pick a name for your system",
    "predictions": [
        {   "train": "what you trained on", # e.g. "X-GENRE-train (train split)"
            "test": "what you evaluated on", # should be "en-ginco"
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
- a table with the results, e.g. `results/results-en-ginco.md`
