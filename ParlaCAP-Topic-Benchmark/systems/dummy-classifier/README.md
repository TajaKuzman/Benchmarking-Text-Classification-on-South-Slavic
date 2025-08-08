# Classification with dummy models

We evaluated the following classifiers:
- Dummy with "stratified" method: DummyClassifier(strategy="stratified"),
- Dummy with "most frequent" method: DummyClassifier(strategy="most_frequent")

The selection of the stratified and most frequent category is based on the ParlaCAP training dataset (29779 instances from ParlaMint datasets) that was used to fine-tune the ParlaCAP classifier and most of the other models that are compared in the benchmark.
