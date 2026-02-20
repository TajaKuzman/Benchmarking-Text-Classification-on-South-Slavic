# Benchmarking Text Classification Tasks on South Slavic languages

The repository comprises code and results for benchmarking non-neural methods, fine-tuned BERT-like models and instruction-tuned GPT models on various benchmarks that comprise South Slavic languages:
- [automatic genre identification in web texts](Genre-Automatic-Identification-Benchmark)
- [news topic classification](IPTC-NewsTopic-Benchmark)
- [topic classification in parliamentary texts](ParlaCAP-Topic-Benchmark)
- [sentiment classification in parliamentary texts](ParlaSent-Benchmark)
- [causal commonsense reasoning (COPA)](DIALECT-COPA-Benchmark)
- [physical commonsense reasoning (PIQA)](CLASSLA-PIQA-Benchmark)

The experiments and results are further discussed in a paper ["State of the Art in Text Classification for South Slavic Languages: Fine-Tuning or Prompting?"](https://arxiv.org/abs/2511.07989) by Kuzman Pungeršek et al. (2026)

**Interactive Dashboard** - see the results and compare the models in our interactive dashboard: https://www.clarin.si/classla-llm-dashboard/

## Results Overview

### Automatic Genre Identification in Web Texts

![](Genre-Automatic-Identification-Benchmark/evaluation-for-the-paper/genre-results-heatmap.png)

![](Genre-Automatic-Identification-Benchmark/evaluation-for-the-paper/gpt_comparison.png)

### News Topic Classification

![](IPTC-NewsTopic-Benchmark/evaluation-for-the-paper/topic-classification-results.png)
![](IPTC-NewsTopic-Benchmark/evaluation-for-the-paper/topic_gpt_comparison.png)

### Topic Classification in Parliamentary Texts

![](ParlaCAP-Topic-Benchmark/evaluation-for-the-paper/parlacap-topic-results-heatmap.png)
![](ParlaCAP-Topic-Benchmark/evaluation-for-the-paper/parlacap_gpt_comparison.png)

### Sentiment Classification in Parliamentary Texts

![](ParlaSent-Benchmark/evaluation-for-the-paper/sentiment-results-heatmap.png)

![](ParlaSent-Benchmark/evaluation-for-the-paper/sentiment_gpt_comparison.png)

### Causal Commonsense reasoning (COPA)

![](DIALECT-COPA-Benchmark/evaluation-for-the-paper/copa-results-heatmap.png)
![](DIALECT-COPA-Benchmark/evaluation-for-the-paper/gpt_comparison.png)

### Physical Commonsense reasoning (PIQA)

![](CLASSLA-PIQA-Benchmark/evaluation-for-the-paper/piqa-results-heatmap.png)
![](CLASSLA-PIQA-Benchmark/evaluation-for-the-paper/gpt_comparison.png)
