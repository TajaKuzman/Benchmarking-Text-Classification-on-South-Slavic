# Other closed-source models

We access the models through the [OpenRouter](https://openrouter.ai/) platform which provides a single API to access various closed-source models, including those from Google, Anthropic, Claude etc.

We evaluate the following models:
- [`google/gemini-2.5-flash`](https://openrouter.ai/google/gemini-2.5-flash)
- [`mistralai/mistral-medium-3.1`](https://openrouter.ai/mistralai/mistral-medium-3.1/api)
- [`anthropic/claude-haiku-4.5`](https://openrouter.ai/anthropic/claude-haiku-4.5)

Later, we will also include more expensive models:
- [`google/gemini-2.5-pro`](https://openrouter.ai/google/gemini-2.5-pro)
- [`anthropic/claude-sonnet-4.5`](https://openrouter.ai/anthropic/claude-sonnet-4.5)

We use the same prompt as for the local and OpenAI models, and use the same code as for the OpenAI models.

`google/gemini-2.5-pro` outputted some empty responses (app. 2-4 per dataset) - the invalid responses were manually replaced with `2` so that accuracy could be calculated (if the invalid response is a string, this introduced problems)