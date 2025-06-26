# OpenAI's GPT models

We evaluate the following models:
- GPT-4o ("gpt-4o-2024-08-06"),
- GPT-3.5-Turbo ("gpt-3.5-turbo-0125"),
- GPT-4o-mini ("gpt-4o-mini-2024-07-18")

We use the following prompt:
```python

label_dict = {'disaster, accident and emergency incident': 0,
 'human interest': 1,
 'politics': 2,
 'education': 3,
 'crime, law and justice': 4,
 'economy, business and finance': 5,
 'conflict, war and peace': 6,
 'arts, culture, entertainment and media': 7,
 'labour': 8,
 'weather': 9,
 'religion': 10,
 'society': 11,
 'health': 12,
 'environment': 13,
 'lifestyle and leisure': 14,
 'science and technology': 15,
 'sport': 16}

reverse_dict = {x[0]:x[1] for x in enumerate(list(label_dict.keys()))}

label_dict_with_description_ext = {
	'disaster, accident and emergency incident - man-made or natural events resulting in injuries, death or damage, e.g., explosions, transport accidents, famine, drowning, natural disasters, emergency planning and response.': 0,
	'human interest - news about life and behavior of royalty and celebrities, news about obtaining awards, ceremonies (graduation, wedding, funeral, celebration of launching something), birthdays and anniversaries, and news about silly or stupid human errors.': 1,
	'politics - news about local, regional, national and international exercise of power, including news about election, fundamental rights, government, non-governmental organisations, political crises, non-violent international relations, public employees, government policies.': 2,
	'education - all aspects of furthering knowledge, formally or informally, including news about schools, curricula, grading, remote learning, teachers and students.': 3,
	'crime, law and justice - news about committed crime and illegal activities, the system of courts, law and law enforcement (e.g., judges, lawyers, trials, punishments of offenders).': 4,
	'economy, business and finance - news about companies, products and services, any kind of industries, national economy, international trading, banks, (crypto)currency, business and trade societies, economic trends and indicators (inflation, employment statistics, GDP, mortgages, ...), international economic institutions, utilities (electricity, heating, waste management, water supply).': 5,
	'conflict, war and peace - news about terrorism, wars, wars victims, cyber warfare, civil unrest (demonstrations, riots, rebellions), peace talks and other peace activities.': 6,
	'arts, culture, entertainment and media - news about cinema, dance, fashion, hairstyle, jewellery, festivals, literature, music, theatre, TV shows, painting, photography, woodworking, art exhibitions, libraries and museums, language, cultural heritage, news media, radio and television, social media, influencers, and disinformation.': 7,
	'labour - news about employment, employment legislation, employees and employers, commuting, parental leave, volunteering, wages, social security, labour market, retirement, unemployment, unions.': 8,
	'weather - news about weather forecasts, weather phenomena and weather warning.': 9,
	'religion - news about religions, cults, religious conflicts, relations between religion and government, churches, religious holidays and festivals, religious leaders and rituals, and religious texts.': 10,
	'society - news about social interactions (e.g., networking), demographic analyses, population census, discrimination, efforts for inclusion and equity, emigration and immigration, communities of people and minorities (LGBTQ, older people, children, indigenous people, etc.), homelessness, poverty, societal problems (addictions, bullying), ethical issues (suicide, euthanasia, sexual behavior) and social services and charity, relationships (dating, divorce, marriage), family (family planning, adoption, abortion, contraception, pregnancy, parenting).': 11,
	'health - news about diseases, injuries, mental health problems, health treatments, diets, vaccines, drugs, government health care, hospitals, medical staff, health insurance.': 12,
	'environment - news about climate change, energy saving, sustainability, pollution, population growth, natural resources, forests, mountains, bodies of water, ecosystem, animals, flowers and plants.': 13,
	'lifestyle and leisure - news about hobbies, clubs and societies, games, lottery, enthusiasm about food or drinks, car/motorcycle lovers, public holidays, leisure venues (amusement parks, cafes, bars, restaurants, etc.), exercise and fitness, outdoor recreational activities (e.g., fishing, hunting), travel and tourism, mental well-being, parties, maintaining and decorating house and garden.': 14,
	'science and technology -  news about natural sciences and social sciences, mathematics, technology and engineering, scientific institutions, scientific research, scientific publications and innovation.': 15,
	'sport - news about sports that can be executed in competitions - basketball, football, swimming, athletics, chess, dog racing, diving, golf, gymnastics, martial arts, climbing, etc.; sport achievements, sport events, sport organisation, sport venues (stadiums, gymnasiums, ...), referees, coaches, sport clubs, drug use in sport.': 16}

	structured_prompt_label_description = f"""
			### Task
			Your task is to classify the provided text into a topic label, meaning that you need to recognize what is the topic of the text. You will be provided with a news text, delimited by single quotation marks. Always provide a label, even if you are not sure.

			### Output format
			Return a valid JSON dictionary with the following key: 'topic' and a value should be an integer which represents one of the labels according to the following dictionary: {label_dict_with_description_ext}.
			"""

	for text in tqdm(texts):
		completion = client.chat.completions.create(model=model,
		response_format={ "type": "json_object"},
		messages=[
		{
			"role": "user",
			"content": prompt + f"\nText: '{text}'"}
		],
		temperature = 0)

```

The evaluation of the three models on the IPTC-test dataset cost 5.77$.