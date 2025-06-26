# OpenAI's GPT models

We evaluate the following models:
- GPT-4o ("gpt-4o-2024-08-06"),
- GPT-3.5-Turbo ("gpt-3.5-turbo-0125"),
- GPT-4o-mini ("gpt-4o-mini-2024-07-18")

We use the following prompt:

```python
	start_time = time.time()

	for text in tqdm(texts):

		sentiment_description = {
			"Negative - text that is entirely or predominantly negative":  0, 
			"Neutral - text that only contains non-sentiment-related statements": 1,
			"Positive - text that is entirely or predominantly positive": 2
		}

		completion = client.chat.completions.create(model="gpt-4o-2024-08-06",
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


The evaluation of the three models on the X-GINCO and EN-GINCO dataset cost 3.27$.