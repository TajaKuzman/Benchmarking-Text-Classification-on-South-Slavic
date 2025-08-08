# OpenAI's GPT models

We evaluate the following models:
- GPT-4o ("gpt-4o-2024-08-06"),
- GPT-3.5-Turbo ("gpt-3.5-turbo-0125"),
- GPT-4o-mini ("gpt-4o-mini-2024-07-18")
- GPT-5 ("gpt-5", evaluated on 8/8/2025)
- GPT-5-nano ("gpt-5-nano-2025-08-07")
- GPT-5-mini ("gpt-5-mini-2025-08-07")

We use the following prompt:

```python
	start_time = time.time()

	for text in tqdm(texts):

		sentiment_description = {
			"Negative - text that is entirely or predominantly negative":  0, 
			"Neutral - text that only contains non-sentiment-related statements": 1,
			"Positive - text that is entirely or predominantly positive": 2
		}

		completion = client.chat.completions.create(model=gpt_model,
		response_format= {"type": "json_object"},
		messages= [
		{
			"role": "user",
			"content": f"""
			### Task
				Your task is to classify the provided parliamentary text into a sentiment label, meaning that you need to recognize whether the speaker's sentiment towards the topic is negative, neutral, positive or somewhere in between. You will be provided with an excerpt from a parliamentary speech in {lang} language, delimited by single quotation marks. Always provide a label, even if you are not sure.


			### Output format
				Return a valid JSON dictionary with the following key: 'sentiment' and a value should be an integer which represents one of the labels according to the following dictionary: {sentiment_description}.

				Text: '{text}'
		"""
	}
		],
		temperature = 0)
```

! Note: the same prompt is used for all models. The only difference is that for the v5 models, only the default temperature = 1 can be used, while we set temperature to 0 for other models (to ensure more deterministic output). The default reasoning effort is used for v5 models - ``medium``, as the parameter reasoning_effort did not work when the models were introduced yet.