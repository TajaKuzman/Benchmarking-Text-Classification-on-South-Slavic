# OpenAI's GPT models

We evaluate the following models:
- GPT-4o ("gpt-4o-2024-08-06"),
- GPT-3.5-Turbo ("gpt-3.5-turbo-0125"),
- GPT-4o-mini ("gpt-4o-mini-2024-07-18")

We use the following prompt:

```python
	# Add dictionary with information on the language and the home country of the dataset
	lang_parl_dict = {
	'hr': {"lang": "Croatian", "parl": "Croatian"},
	#'SI': {"lang": "Slovenian", "parl": "Slovenian"},
	#'RS': {"lang": "Serbian", "parl": "Serbian"},
	'en': {"lang": "English", "parl": "British"},
	#'BA': {"lang": "Bosnian", "parl": "Bosnian"},
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
