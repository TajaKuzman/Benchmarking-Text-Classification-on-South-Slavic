# Benchmarking LLMs on commonsense reasoning COPA datasets for South Slavic languages and dialects

The code for all evaluated models is available in the [systems](systems) directory.

## Benchmark scores

![](evaluation-for-the-paper/copa-results-heatmap.png)
![](evaluation-for-the-paper/gpt_comparison.png)

| Model               |   English (accuracy) |   Slovenian (accuracy) |   Croatian (accuracy) |   Serbian (accuracy) |   Macedonian (accuracy) |   Cerkno Dialect (accuracy) |   Chakavian Dialect (accuracy) |   Torlak Dialect (accuracy) |
|:--------------------|---------------------:|-----------------------:|----------------------:|---------------------:|------------------------:|----------------------------:|-------------------------------:|----------------------------:|
| GPT-5               |                0.996 |                  0.998 |                 0.99  |              nan     |                   0.986 |                     nan     |                          0.916 |                     nan     |
| Gemini 2.5 Flash    |                0.99  |                  0.974 |                 0.98  |                0.972 |                   0.968 |                       0.742 |                          0.79  |                       0.944 |
| GPT-4o              |                0.988 |                  0.97  |                 0.972 |                0.972 |                   0.968 |                       0.676 |                          0.774 |                       0.932 |
| Claude Haiku 4.5    |                0.986 |                  0.926 |                 0.948 |                0.956 |                   0.924 |                       0.562 |                          0.706 |                       0.862 |
| Mistral Medium 3.1  |                0.986 |                  0.9   |                 0.942 |                0.932 |                   0.9   |                       0.532 |                          0.7   |                       0.824 |
| LLaMA 3.3           |                0.986 |                  0.87  |                 0.926 |                0.922 |                   0.894 |                       0.536 |                          0.674 |                       0.798 |
| Qwen 3              |                0.972 |                  0.826 |                 0.9   |                0.89  |                   0.862 |                       0.544 |                          0.59  |                       0.782 |
| Gemma 3             |                0.97  |                  0.862 |                 0.894 |                0.9   |                   0.904 |                       0.578 |                          0.642 |                       0.82  |
| GPT-3.5-Turbo       |                0.952 |                  0.842 |                 0.854 |                0.808 |                   0.776 |                       0.528 |                          0.612 |                       0.734 |
| GaMS-27B            |                0.864 |                  0.832 |                 0.788 |                0.784 |                   0.692 |                       0.586 |                          0.592 |                       0.636 |
| DeepSeek-R1-Distill |                0.748 |                  0.53  |                 0.57  |                0.554 |                   0.546 |                       0.492 |                          0.504 |                       0.514 |
| Dummy (Frequent)    |                0.5   |                  0.5   |                 0.5   |                0.5   |                   0.5   |                       0.5   |                          0.5   |                       0.5   |

## Datasets

Dialectal test data (the DIALECT-COPA collection) are available at a private GitHub repository (https://github.com/clarinsi/dialect-copa-test) - access can be obtained upon request to Nikola Ljubešić (nikola.ljubesic@ijs.si) or Taja Kuzman Pungeršek (taja.kuzman@ijs.si).

Standard test data are available in the CLARIN.SI repository: [COPA-HR](http://hdl.handle.net/11356/1404), [COPA-SR](
http://hdl.handle.net/11356/1708) and [COPA-MK](http://hdl.handle.net/11356/1687).

## Prompt

```python

    for line in open('test.jsonl'):
        entry=json.loads(line)

		if df_test_name != "copa-en":
			prompt= 'You will be given a task. The task definition is in English, but the task itself is in another language. Here is the task!\nGiven the premise "'+entry['premise']+'",'
			if entry['question']=='cause':
				prompt+=' and that we are looking for the cause of this premise,'
			else:
				prompt+=' and that we are looking for the result of this premise, '
			prompt+=f"""which hypothesis is more plausible?\nHypothesis 1: "{entry['choice1']}".\nHypothesis 2: "{entry['choice2']}".
					
			### Output format
				Return a valid JSON dictionary with the following key: 'answer' and a value should be an integer -- either 1 (if hypothesis 1 is more plausible) or 2 (if hypothesis 2 is more plausible).
			"""
		elif df_test_name == "copa-en":
			prompt= 'You will be given a task. The task definition is in English, as is the task itself. Here is the task!\nGiven the premise "'+entry['premise']+'",'
			if entry['question']=='cause':
				prompt+=' and that we are looking for the cause of this premise,'
			else:
				prompt+=' and that we are looking for the result of this premise,'
			prompt+=f"""which hypothesis is more plausible?\nHypothesis 1: "{entry['choice1']}".\nHypothesis 2: "{entry['choice2']}".
					
			### Output format
				Return a valid JSON dictionary with the following key: 'answer' and a value should be an integer -- either 1 (if hypothesis 1 is more plausible) or 2 (if hypothesis 2 is more plausible).
			"""

    completion = client.chat.completions.create(model=args.model,
    messages=[
    {
        "role": "user",
        "content": prompt}
    ],
    temperature = 0)
    )
    response=completion.choices[0].message.content
```

## Contributing to the benchmark

Should you wish to contribute an entry, feel free to submit a folder in the [systems](systems) directory with or without the code used (see the submission examples in the directory).

The results JSON file name should start with `submission-` and the content should be structured like this:

```python
{
	"system": "Pick a name for your system",
	"predictions": [
		{   "train": "what you trained on",
			"test": "what you evaluated on", # should be "copa-en", "copa-sl", "copa-hr-ckm", "copa-hr", "copa-mk", "copa-sl-cer", "copa-sr" or "copa-sr-tor"
			"predictions": [....] # The length of predictions should match the length of test data
		},
	],
	# Additional information, e.g. fine-tuning params:
	"model": "EMBEDDIA/crosloengual-bert",
	"lr": "4e-5",
	"epoch": "15"
}
```

All submission JSON files should be saved in a `submissions` directory inside the directory for your system. They will be evaluated against the datasets in the `datasets` directory.

It is highly encouraged that you also provide additional information about your system in a README file, and that you provide the code used for the classification with the system.

## Evaluation

Micro and Macro F1 scores will be used to evaluate and compare systems.

The submissions are evaluated using the following code with the path to the submissions directory (e.g., ``systems/dummy-classifier/submissions``) as the argument. The log file is to be saved in the relevant system directory:
```python eval.py "submission-path" > systems/dummy-classifier/evaluation.log```

The code produces:
- a JSON file with the results of all tested models: `results/results.json`
- a table with the results, e.g. `results/results-copa-hr.md`


