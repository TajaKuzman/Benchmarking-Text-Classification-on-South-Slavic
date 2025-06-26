# Classification with Non-Neural Classifiers

We train the models on the ParlaSent dataset (13000 instances). The texts are represented using the TD-IDF representation (created with TfidfVectorizer from the Scikit-Learn library).

We use the following models, provided through the Scikit-Learn library:
-  Naive Bayes Classifier: This probabilistic machine learning algorithm learns the statistical relationships between the words present in the documents, taking into account also their frequency, and the corresponding sentiment categories. We use the Complement Naive Bayes implementation, which is especially suitable for imbalanced multi-class datasets. 
-  Support Vector Machine (SVM): the SVM model is a linear classifier that determines the boundaries between classes in form of a separating hyperplane. Its efficacy is particularly notable in high-dimensional spaces, making it highly applicable in the context of text categorization tasks, where the feature set can encompass the entire dataset vocabulary. In this study, we employ the SVC implementation with the linear kernel, which supports multi-class categorization.

Hyperparameters:
- Naive Bayes: ComplementNB model with the default hyperparameters.
- SVM: SVC model with the linear kernel and the regularization parameter C set to 2.

Code: *non-neural-classifiers.ipynb*