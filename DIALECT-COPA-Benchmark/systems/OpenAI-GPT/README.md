# OpenAI's GPT models

We evaluate the following models:
- GPT-4o ("gpt-4o-2024-08-06"),
- GPT-3.5-Turbo ("gpt-3.5-turbo-0125"),
- GPT-4o-mini ("gpt-4o-mini-2024-07-18")
- GPT-5 ("gpt-5", evaluated on 8/8/2025)
- GPT-5-nano ("gpt-5-nano-2025-08-07")
- GPT-5-mini ("gpt-5-mini-2025-08-07")

! Note: the same prompt is used for all models. The only difference is that for the v5 models, only the default temperature = 1 can be used, while we set temperature to 0 for other models (to ensure more deterministic output). The default reasoning effort is used for v5 models - ``medium``, as the parameter reasoning_effort did not work when the models were introduced yet.