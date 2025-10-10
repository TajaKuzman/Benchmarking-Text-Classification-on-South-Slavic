# Other closed-source models

We access the models through the [OpenRouter](https://openrouter.ai/) platform which provides a single API to access various closed-source models, including those from Google, Anthropic, Claude etc.

We evaluate the following models:
- [`google/gemini-2.5-flash-lite`](https://openrouter.ai/google/gemini-2.5-flash-lite/api)
- [`google/gemini-2.5-flash`](https://openrouter.ai/google/gemini-2.5-flash)
- [`mistralai/mistral-medium-3.1`](https://openrouter.ai/mistralai/mistral-medium-3.1/api)
- [`mistralai/mistral-small-3.2-24b-instruct`](https://openrouter.ai/mistralai/mistral-small-3.2-24b-instruct)
- [`cohere/command-a`](https://openrouter.ai/cohere/command-a)

Anthropic models were not included, because they do not enable the `response_format` parameter.
We don't include [`google/gemini-2.5-pro`](https://openrouter.ai/google/gemini-2.5-pro) because it is too expensive (0.01€ per instance).

We use the same prompt as for the local and OpenAI models, and use the same code as for the OpenAI models:
```python
	lang_parl_dict = {
	'HR': {"lang": "Croatian", "parl": "Croatian"},
	'RS': {"lang": "Serbian", "parl": "Serbian"},
	'GB': {"lang": "English", "parl": "British"},
	'BA': {"lang": "Bosnian", "parl": "Bosnian"},
	}

	start_time = time.time()

	for i in list(zip(texts, langs)):
		text = i[0]
		lang = i[1]
		parl = lang_parl_dict[lang]["parl"]
		language = lang_parl_dict[lang]["lang"]

		completion = client.chat.completions.create(model=gpt_model,
		response_format= {"type": "json_object"},
		messages= [
		{
			"role": "user",
			"content": f"""
			### Task
				Your task is to classify the provided text into a policy agenda topic label, meaning that you need to recognize what is the predominant topic of the text. You will be provided with an excerpt from a parliamentary speech from the {parl} parliament in {language} language, delimited by single quotation marks. Always provide a label, even if you are not sure.


				Follow the following rule: if the speech mentions a policy area and a policy instrument (e.g., taxes, laws), pick the label based on the area, not the instrument (e.g., annotate mortgage tax changes with 14 (Housing), law on education with 6 (Education)).


			### Output format
				Return a valid JSON dictionary with the following key: 'topic' and a value should be an integer which represents one of the labels according to the following dictionary: {majortopics_description}.

				
				Text: '{text}
		"""
	}
		],
		temperature = 0)
```