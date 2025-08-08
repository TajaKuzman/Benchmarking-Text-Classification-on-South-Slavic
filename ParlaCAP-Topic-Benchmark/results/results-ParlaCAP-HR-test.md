## ParlaCAP-HR-test

| Model                  | Test Dataset     |   Macro F1 |   Micro F1 |
|:-----------------------|:-----------------|-----------:|-----------:|
| ParlaCAP-classifier    | ParlaCAP-HR-test |      0.681 |      0.669 |
| gpt-4o-2024-08-06      | ParlaCAP-HR-test |      0.67  |      0.664 |
| gpt-4o-mini-2024-07-18 | ParlaCAP-HR-test |      0.584 |      0.582 |
| llama3.3:latest        | ParlaCAP-HR-test |      0.574 |      0.579 |
| gemma3:27b             | ParlaCAP-HR-test |      0.561 |      0.557 |
| gpt-3.5-turbo-0125     | ParlaCAP-HR-test |      0.493 |      0.493 |
| deepseek-r1:14b        | ParlaCAP-HR-test |      0.291 |      0.26  |
| SVC                    | ParlaCAP-HR-test |      0.073 |      0.125 |
| dummy-stratified       | ParlaCAP-HR-test |      0.054 |      0.061 |
| COMPLEMENTNB           | ParlaCAP-HR-test |      0.02  |      0.105 |
| dummy-most_frequent    | ParlaCAP-HR-test |      0.006 |      0.067 |