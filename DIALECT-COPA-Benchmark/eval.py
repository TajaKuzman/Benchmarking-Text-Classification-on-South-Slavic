gold=[int(e) for e in open('test_labels.txt')]

truth=[] # whether the prediction is identical to gold
valid=[] # whether the prediction is a valid numerical value
for idx,response in enumerate(responses):
    try:
        label=int(response.strip('"'))-1
    except:
        truth.append(False)
        valid.append(False)
        continue
      truth.append(gold[idx]==label)
      valid.append(True)

print(sum(truth.values())/len(truth)) # accuracy
print(sum(valid.values())/len(valid)) # valid response rate
