# Classification with dummy models

We evaluated the following classifiers:
- Dummy with "stratified" method: DummyClassifier(strategy="stratified"),
- Dummy with "most frequent" method: DummyClassifier(strategy="most_frequent")

The selection of the stratified and most frequent category is based on the training split of the [EMMediaTopic dataset](https://www.clarin.si/repository/xmlui/handle/11356/1991) that was used to fine-tune the IPTC classifier and other models that are compared in the benchmark.
