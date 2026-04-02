# Training Runs Table

| Candidate | MLflow run link | Code version (git sha) | Key hyperparams | Key model metrics | Key training cost metrics | Notes |
|---|---|---|---|---|---|---|
| baseline |  |  | classifier=logreg, word_ngram=1-2, char_tfidf=false | val_macro_f1=, test_macro_f1=, test_top3_accuracy= | train_wall_sec=, model_size_mb=, peak_ram_mb= | simplest baseline |
| v1 |  |  | classifier=logreg, word_ngram=1-2, char_tfidf=true | val_macro_f1=, test_macro_f1=, test_top3_accuracy= | train_wall_sec=, model_size_mb=, peak_ram_mb= | richer text features |
| v2 |  |  | classifier=linearsvc, word_ngram=1-2 | val_macro_f1=, test_macro_f1=, test_top3_accuracy= | train_wall_sec=, model_size_mb=, peak_ram_mb= | fast linear margin model |
| v3 |  |  | classifier=sgd, word_ngram=1-2 | val_macro_f1=, test_macro_f1=, test_top3_accuracy= | train_wall_sec=, model_size_mb=, peak_ram_mb= | low-cost candidate |

## Recommended candidates
- Best quality:
- Best speed/cost tradeoff:
