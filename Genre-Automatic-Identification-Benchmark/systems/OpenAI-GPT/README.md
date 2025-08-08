# OpenAI's GPT models

We evaluate the following models:
- GPT-4o ("gpt-4o-2024-08-06"),
- GPT-3.5-Turbo ("gpt-3.5-turbo-0125"),
- GPT-4o-mini ("gpt-4o-mini-2024-07-18")
- GPT-5 ("gpt-5", evaluated on 8/8/2025)
- GPT-5-nano ("gpt-5-nano-2025-08-07")
- GPT-5-mini ("gpt-5-mini-2025-08-07")

We use the following prompt:
```json
			messages= [
			{
				"role": "user",
				"content": f"""
				### Task
				Your task is to classify the following text according to genre. Genres are text types, defined by the function of the text, author’s purpose and form of the text. Always provide a label, even if you are not sure.

				### Output format
					Return a valid JSON dictionary with the following key: 'genre' and a value should be an integer which represents one of the labels according to the following dictionary: {labels_dict}.

					
					Text: '{text}'
			"""
				}
```

! Note: the same prompt is used for all models. The only difference is that for the v5 models, only the default temperature = 1 can be used, while we set temperature to 0 for other models (to ensure more deterministic output). The default reasoning effort is used for v5 models - ``medium``, as the parameter reasoning_effort did not work when the models were introduced yet.

The evaluation of the GPT-4o, GPT-3.5-Turbo and GPT-4o-mini models together on the X-GINCO and EN-GINCO dataset cost $3.27.

The evaluation of GPT-5, GPT-5-mini and GPT-5-nano models together on the EN-GINCO dataset (272 instances) cost $0.86. GPT-5-nano took 12 minutes, GPT-5-mini took 11 minutes, and GPT-5 took 20 minutes to classify 272 instances.

The evaluation of GPT-5, GPT-5-mini and GPT-5-nano models together on the X-GINCO dataset (790 instances) cost $2.30. GPT-5-nano took 35 minutes, GPT-5-mini took 37 minutes, and GPT-5 took 54 minutes to classify app. 800 instances.