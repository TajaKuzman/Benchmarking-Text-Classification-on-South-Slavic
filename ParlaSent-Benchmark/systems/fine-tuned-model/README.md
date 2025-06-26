# Prediction with the fine-tuned XLM-R-ParlaSent classifier

The classifier is based on [XLM-R-Parla](https://huggingface.co/classla/xlm-r-parla) model and it was fine-tuned on the [ParlaSent dataset](http://hdl.handle.net/11356/1868).

It is available on Hugging Face: https://huggingface.co/classla/xlm-r-parlasent

This is a regression model, but we map the float predictions to the 3-label schema (according to the mapping published in the model card on HuggingFace).

The details on the model development are provided in the paper [The ParlaSent multilingual training dataset for sentiment identification in parliamentary proceedings](http://www.lrec-conf.org/proceedings/lrec-coling-2024/pdf/2024.main-1.1393.pdf) (Mochtak et al., 2023).