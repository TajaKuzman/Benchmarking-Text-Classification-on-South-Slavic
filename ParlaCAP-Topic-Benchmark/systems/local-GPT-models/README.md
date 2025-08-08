# Evaluation of local GPT models

The following GPT models were installed locally and ran through the [Ollama API service](https://ollama.com/). We used the quantized models as provided through the [Ollama library](https://ollama.com/library/).

The following models were evaluated:
- [Gemma 3 (27B model)](https://ollama.com/library/gemma3) `gemma3:27b`
- [DeepSeek-R1 (14B model)](https://ollama.com/library/deepseek-r1) `deepseek-r1:14b`
- [Llama 3.3 model (70B model)](https://ollama.com/library/llama3.3) `llama3.3:latest`

The prompt:
```python

# define label description and label dictionary

labels_dict = {
 1: 'Macroeconomics',
 2: 'Civil Rights',
 3: 'Health',
 4: 'Agriculture',
 5: 'Labor',
 6: 'Education',
 7: 'Environment',
 8: 'Energy',
 9: 'Immigration',
 10: 'Transportation',
 12: 'Law and Crime',
 13: 'Social Welfare',
 14: 'Housing',
 15: 'Domestic Commerce',
 16: 'Defense',
 17: 'Technology',
 18: 'Foreign Trade',
 19: 'International Affairs',
 20: 'Government Operations',
 21: 'Public Lands',
 23: 'Culture',
 0: 'Other',
 }

majortopics_description = {
 'Macroeconomics - issues related to domestic macroeconomic policy, such as the state and prospect of the national economy, economic policy, inflation, interest rates, monetary policy, cost of living, unemployment rate, national budget, public debt, price control, tax enforcement, industrial revitalization and growth.': 1,
 'Civil Rights - issues related to civil rights and minority rights, discrimination towards races, gender, sexual orientation, handicap, and other minorities, voting rights, freedom of speech, religious freedoms, privacy rights, protection of personal data, abortion rights, anti-government activity groups (e.g., local insurgency groups), religion and the Church.': 2,
 'Health - issues related to health care, health care reforms, health insurance, drug industry, medical facilities, medical workers, disease prevention, treatment, and health promotion, drug and alcohol abuse, mental health, research in medicine, medical liability and unfair medical practices.': 3,
 'Agriculture - issues related to agriculture policy, fishing, agricultural foreign trade, food marketing, subsidies to farmers, food inspection and safety, animal and crop disease, pest control and pesticide regulation, welfare for animals in farms, pets, veterinary medicine, agricultural research.': 4,
 'Labor - issues related to labor, employment, employment programs, employee benefits, pensions and retirement accounts, minimum wage, labor law, job training, labor unions, worker safety and protection, youth employment and seasonal workers.': 5,
 'Education - issues related to educational policies, primary and secondary schools, student loans and education finance, the regulation of colleges and universities, school reforms, teachers, vocational training, evening schools, safety in schools, efforts to improve educational standards, and issues related to libraries, dictionaries, teaching material, research in education.': 6,
 'Environment - issues related to environmental policy, drinking water safety, all kinds of pollution (air, noise, soil), waste disposal, recycling, climate change, outdoor environmental hazards (e.g., asbestos), species and forest protection, marine and freshwater environment, hunting, regulation of laboratory or performance animals, land and water resource conservation, research in environmental technology.': 7,
 'Energy - issues related to energy policy, electricity, regulation of electrical utilities, nuclear energy and disposal of nuclear waste, natural gas and oil, drilling, oil spills, oil and gas prices, heat supply, shortages and gasoline regulation, coal production, alternative and renewable energy, energy conservation and energy efficiency, energy research.': 8,
 'Immigration - issues related to immigration, refugees, and citizenship, integration issues, regulation of residence permits, asylum applications; criminal offences and diseases caused by immigration.': 9,
 'Transportation - issues related to mass transportation construction and regulation, bus transport, regulation related to motor vehicles, road construction, maintenance and safety, parking facilities, traffic accidents statistics, air travel, rail travel, rail freight, maritime transportation, inland waterways and channels, transportation research and development.': 10,
 'Law and Crime - issues related to the control, prevention, and impact of crime; all law enforcement agencies, including border and customs, police, court system, prison system; terrorism, white collar crime, counterfeiting and fraud, cyber-crime, drug trafficking, domestic violence, child welfare, family law, juvenile crime.': 12,
 'Social Welfare - issues related to social welfare policy, the Ministry of Social Affairs, social services, poverty assistance for low-income families and for the elderly, parental leave and child care, assistance for people with physical or mental disabilities, including early retirement pension, discounts on public services, volunteer associations (e.g., Red Cross), charities, and youth organizations.': 13,
 'Housing - issues related to housing, urban affairs and community development, housing market, property tax, spatial planning, rural development, location permits, construction inspection, illegal construction, industrial and commercial building issues, national housing policy, housing for low-income individuals, rental housing, housing for the elderly, e.g., nursing homes, housing for the homeless and efforts to reduce homelessness, research related to housing.': 14,
 'Domestic Commerce - issues related to banking, finance and internal commerce, including stock exchange, investments, consumer finance, mortgages, credit cards, insurance availability and cost, accounting regulation, personal, commercial, and municipal bankruptcies, programs to promote small businesses, copyrights and patents, intellectual property, natural disaster preparedness and relief, consumer safety; regulation and promotion of tourism, sports, gambling, and personal fitness; domestic commerce research.': 15,
 'Defense - issues related to defense policy, military intelligence, espionage, weapons, military personnel, reserve forces, military buildings, military courts, nuclear weapons, civil defense, including firefighters and mountain rescue services, homeland security, military aid or arms sales to other countries, prisoners of war and collateral damage to civilian populations, military nuclear and hazardous waste disposal and military environmental compliance, defense alliances and agreements, direct foreign military operations, claims against military, defense research.': 16,
 'Technology - issues related to science and technology transfer and international science cooperation, research policy, government space programs and space exploration, telephones and telecommunication regulation, broadcast media (television, radio, newspapers, films), weather forecasting, geological surveys, computer industry, cyber security.': 17,
 'Foreign Trade - issues related to foreign trade, trade negotiations, free trade agreements, import regulation, export promotion and regulation, subsidies, private business investment and corporate development, competitiveness, exchange rates, the strength of national currency in comparison to other currencies, foreign investment and sales of companies abroad.': 18,
 'International Affairs - issues related to international affairs, foreign policy and relations to other countries, issues related to the Ministry of Foreign Affairs, foreign aid, international agreements (such as Kyoto agreement on the environment, the Schengen agreement), international organizations (including United Nations, UNESCO, International Olympic Committee, International Criminal Court), NGOs, issues related to diplomacy, embassies, citizens abroad; issues related to border control; issues related to international finance, including the World Bank and International Monetary Fund, the financial situation of the EU; issues related to a foreign country that do not impact the home country; issues related to human rights in other countries, international terrorism.': 19,
 'Government Operations - issues related to general government operations, the work of multiple departments, public employees, postal services, nominations and appointments, national mints, medals, and commemorative coins, management of government property, government procurement and contractors, public scandal and impeachment, claims against the government, the state inspectorate and audit, anti-corruption policies, regulation of political campaigns, political advertising and voter registration, census and statistics collection by government; issues related to local government, capital city and municipalities, including decentralization; issues related to national holidays.': 20,
 'Public Lands - issues related to national parks, memorials, historic sites, and protected areas, including the management and staffing of cultural sites; museums; use of public lands and forests, establishment and management of harbors and marinas; issues related to flood control, forest fires, livestock grazing.': 21,
 'Culture - issues related to cultural policies, Ministry of Culture, public spending on culture, cultural employees, issues related to support of theatres and artists; allocation of funds from the national lottery, issues related to cultural heritage': 23,
 'Other - other topics not mentioning policy agendas, including the procedures of parliamentary meetings, e.g., points of order, voting procedures, meeting logistics; interpersonal speech, e.g., greetings, personal stories, tributes, interjections, arguments between the members; rhetorical speech, e.g., jokes, literary references.': 0
 }


	for i in list(zip(texts, langs)):
		text = i[0]
		lang = i[1]
		parl = lang_parl_dict[lang]["parl"]
		language = lang_parl_dict[lang]["lang"]

		current_prompt = f"""
			### Task
				Your task is to classify the provided text into a policy agenda topic label, meaning that you need to recognize what is the predominant topic of the text. You will be provided with an excerpt from a parliamentary speech from the {parl} parliament in {language} language, delimited by single quotation marks. Always provide a label, even if you are not sure.


				Follow the following rule: if the speech mentions a policy area and a policy instrument (e.g., taxes, laws), pick the label based on the area, not the instrument (e.g., annotate mortgage tax changes with 14 (Housing), law on education with 6 (Education)).


			### Output format
				Return a valid JSON dictionary with the following key: 'topic' and a value should be an integer which represents one of the labels according to the following dictionary: {majortopics_description}.

				
				Text: '{text}
		"""
```

In some (rare) cases, the models' output was invalid, e.g. `{"topic": -1}`. These cases were manually changed to `Mix`.