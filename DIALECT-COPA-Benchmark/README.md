Dialectal test data are available at: https://github.com/clarinsi/dialect-copa-test

Standard test data are available in the repository.

```

    for line in open('test.jsonl'):
        entry=json.loads(line)

    prompt= 'You will be given a task. The task definition is in English, but the task itself is in another language. Here is the task!\nGiven the premise "'+entry['premise']+'",'
    if entry['question']=='cause':
        prompt+=' and that we are looking for the cause of this premise,'
    else:
        prompt+=' and that we are looking for the result of this premise,'
    prompt+=' which hypothesis is more plausible?\nHypothesis 1: "'+entry['choice1']+'".\nHypothesis 2: "'+entry['choice2']+'".\nAnswer only with "1" or "2".\nAnswer: '



    completion = client.chat.completions.create(model=args.model,
    messages=[
    {
        "role": "user",
        "content": prompt}
    ],
    temperature = 0)
    )
    response=completion.choices[0].message.content
```


