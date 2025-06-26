# Prediction with the fine-tuned IPTC classifier

The IPTC classifier is based on XLM-RoBERTa-large model and it was fine-tuned on the training split of the [EMMediaTopic dataset](https://www.clarin.si/repository/xmlui/handle/11356/1991), more precisely, on a stratified split containing 15,000 instances in 4 languages.

It is available on Hugging Face: https://huggingface.co/classla/multilingual-IPTC-news-topic-classifier

The development and evaluation of the model is described in the paper [LLM Teacher-Student Framework for Text Classification With No Manually Annotated Data: A Case Study in IPTC News Topic Classification](https://doi.org/10.1109/ACCESS.2025.3544814) (Kuzman and Ljubešić, 2025)