# Evaluation of local GPT models

The following GPT models were installed locally and ran through the [Ollama API service](https://ollama.com/). We used the quantized models as provided through the [Ollama library](https://ollama.com/library/).

The following models were evaluated:
- [Gemma 3 (27B model)](https://ollama.com/library/gemma3) `gemma3:27b`
- [DeepSeek-R1 (14B model)](https://ollama.com/library/deepseek-r1) `deepseek-r1:14b`
- [Llama 3.3 model (70B model)](https://ollama.com/library/llama3.3) `llama3.3:latest`
- [Llama 4 Scout model (17B active parameters, 109B total parameters, 16 experts)](https://ollama.com/library/llama4) `llama4:scout`
- [Qwen 3 (32B)](https://ollama.com/library/qwen3) `qwen3:32b`

The prompt:
```python

	labels_dict = {0: "Negative", 1: "Neutral", 2: "Positive"}

	sentiment_description = {
		"Negative - text that is entirely or predominantly negative":  0, 
		"Neutral - text that only contains non-sentiment-related statements": 1,
		"Positive - text that is entirely or predominantly positive": 2
	}

	for text in texts:
		current_prompt = f"""
			### Task
				Your task is to classify the provided parliamentary text into a sentiment label, meaning that you need to recognize whether the speaker's sentiment towards the topic is negative, neutral, positive or somewhere in between. You will be provided with an excerpt from a parliamentary speech in {lang} language, delimited by single quotation marks. Always provide a label, even if you are not sure.


			### Output format
				Return a valid JSON dictionary with the following key: 'sentiment' and a value should be an integer which represents one of the labels according to the following dictionary: {sentiment_description}.

				Text: '{text}'
		"""

```

In some (rare) cases, the models' output was invalid, e.g. `{"sentiment": -1}`. These cases were manually changed to `Mix`.