# Classification with dummy models

We evaluated the following classifiers:
- Dummy with "stratified" method: DummyClassifier(strategy="stratified"),
- Dummy with "most frequent" method: DummyClassifier(strategy="most_frequent")

The selection of the stratified and most frequent category is based on the [ParlaSent dataset](http://hdl.handle.net/11356/1868) (13000 instances) that was used to fine-tune the XLM-R-parlasent classifier and most of the other models that are compared in the benchmark.
