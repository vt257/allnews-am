# allnews-am
Natural Language Processing for Armenian News Summarization.

## Requirements
- Set up environment variables `ALLNEWS_AM_MYSQL_HOST`, `ALLNEWS_AM_MYSQL_NAME`, `ALLNEWS_AM_MYSQL_USER`, 
`ALLNEWS_AM_MYSQL_PASS` for database access.

## Setup
1. Run `pip install -r requirements.txt`
1. Run `pip install .`

## Word embeddings 
### Training
A corpus of 100k processed news articles can be found 
[here [58MB]](https://storage.googleapis.com/allnews_am/corpus_100k.zip). 
This is produces by the `generate_corpus.py` script.
Download it, unzip it, and put the `corpus_100k` file it in the `data` folder. 
After this, you should be able to run a word2vec training using e.g.
```
python scripts/train_word2vec.py --epochs=5 --corpus=data/corpus_100k
```
which will save the model files in the `models` folder after
the training is complete. Similarly, you can train FastText using
```
python scripts/train_fasttext.py --epochs=5  --corpus=data/corpus_100k
```
You can also run
```
python scripts/train_word2vec.py --help
```
for more training options.

### Evaluation
You can run a basic evaluation on word analogies using
```
python scripts/explore_embeddings.py --model_name=embeddings.model
```
Analogy scores on the `coarse_avetisyan_ghukasyan_analogies.txt` are
shown below, combined with results of Named Entity Recognition (see next section)
trained using that particular set of pre-trained embeddings and a fixed default
architecture. 

For models trained on the 100k news corpus with about 25m tokens:

| Model | Semantic (1858) | Syntactic (5069) | Total (6927) | NER_f1 (test) |
| ------------------------------------------ | --- | ---- | ---- | ----- |
| fastText_sg_100_lr0025_ep30                | 4.0 | 50.0 | 37.7 | 70.21 |
| word2vec_sg_100_lr0025_ep30                | 9.1 | 4.8  | 5.9  | -     |
| ft_sg_100_100k_parag_per_line_cn25         | 4.2 | 51.0 | 38.5 | 70.24 |
| ft_sg_100_100k_sent_per_line_cn25          | 3.9 | 51.6 | 38.8 | 70.06 |
| ft_sg_100_100k_sent_per_line_cn26          | 4.1 | 51.4 | 38.7 | 69.48 |
| ft_sg_100_100k_sent_per_line_cn25_ss00001  | 5.7 | 55.1 | 41.8 | 70.82 |
| ft_sg_100_100k_sent_per_line_cn25_ss000003 | 5.5 | 56.1 | 42.5 | 70.10 |
| ft_sg_100_100k_sent_per_line_cn25_ss000001 | 5.2 | 56.9 | 43.0 | 70.01 |

The NER f1 is the micro averaged f1 score on the test set of pioNER dataset, using
Bi-LSTM (size 50) + Char-Bi-LSTM (embedding 10 / size 20) + Softmax
architecture with 0.5 recurrent dropout for both LSTMs and 0.3
SpatialDropout1D after concatenating word and character embeddings.

Below are the same results for embeddings obtained using the full 1679k news corpus,
which has about 325m tokens:

| Model | Semantic (3392) | Syntactic (7149) | Total (10541) | NER_f1 (train) | NER_f1 (dev) | NER_f1 (test) |
| ---------------------------------------------- | -------- | -------- | -------- | ----- | ----- | ----- |
| fastText_sg_200_lr01_ep10_cn35                 | 0.5      | 46.8     | 31.9     | -     |       |       |
| fastText_sg_100_lr0025_ep5_cn35                | 2.8      | 45.6     | 31.8     | -     |       |       |
| fastText_sg_100_lr005_cn35                     | 1.3      | 39.9     | 27.5     | -     |       |       |
| fastText_sg_100_lr0025_cn35                    | 7.8      | 48.8     | 35.6     | 82.59 | 76.23 | 72.98 |
| fastText_sg_100_lr0025_cn25                    | 5.1      | 51.8     | 36.8     | 81.68 | 76.16 | 72.65 |
| fastText_sg_100_lr0025_cn25_shuff              | 6.1      | 52.1     | 37.3     | 82.75 | 76.97 | 73.27 |
| **fastText_sg_100_lr0025_cn25_shuff_ss000001** | **18.9** | **56.7** | **44.6** | 85.23 | 77.79 | 71.86 |
| fastText_sg_100_lr001_cn25_shuff               | 3.2      | 48.5     | 33.9     | -     |       |       |

Some of the important hyperparameters are in the model names. All models are
trained for 30 epochs unless otherwise specified with ep key. cn stands for 
character n-gram size. shuff uses a shuffled corpus. The downsampling parameter 
is 0.003 unless specified otherwise with ss key.

The embeddings can be improved a little further by adding hywiki (Armenian Wikipedia)
corpus (data dump from 2020-01-20). The results on the combined, shuffled
corpus of 1679k news + Wikipedia (a total of ~380m tokens) are below.

| Model | Semantic (4154) | Syntactic (7203) | Total (11357) | NER_f1 (train) | NER_f1 (dev) | NER_f1 (test) |
| ------------------------------ | ---- | ---- | ---- | ------| ----- | ----- |
| ft_sg_100_lr0025_cn25_ss000001 | 18.9 | 57.0 | 43.0 | 84.81 | 79.01 | 72.52 |

All models are trained for 30 epochs, window 5 and min_count of 5, and 10
negative samples for the loss function.

**NOTE** We are not lowercasing words and we keep the punctuation,
so the scores on the analogy tasks are expected to be worse in comparison
to text pre-processing that performs these steps.

## Named Entity Recognition
### Training
To train a [Bi-LSTM + (char)Bi-LSTM + Softmax] model using the dataset from
[pioNER](https://github.com/ispras-texterra/pioner) you can execute
```
python scripts\train_ner.py
```
and 
```
python scripts\train_ner.py --help
```
for additional options. 

Typical scores using pre-trained fastText embeddings and the default model
parameters are below

|       | precision | recall | f1-score | support |
| ----- | --------- | ------ | -------- | ------- | 
| Train |   0.8348  | 0.8103 |  0.8205  |  130719 |
| Dev   |   0.7772  | 0.7502 |  0.7603  |  32528  |
| Test  |   0.7919  | 0.6554 |  0.7021  |  53606  |

These scores are per-token rather than per-NE, similar to the
baselines in the original [pioNER paper](https://arxiv.org/abs/1810.08699).

### Hyperparameter tuning
The hyperparameters are tuned using the `ft_sg_100_lr0025_cn25_ss000001`
model, which is trained on all news + hywiki. This is a simple exploration
and hyperparameters are tuned one at a time.

| Model                             | Train | Dev   | Test  |
| --------------------------------- | ----- | ----- | ----- |
| Default                           | 84.81 | 79.01 | 72.52 |
| SpatialDropout1D -> Dropout       | 85.37 | 78.92 | 70.81 |
| SpatialDropout1D 0.3 -> 0.5       | 81.44 | 76.72 | 70.57 |
| SpatialDropout1D 0.3 -> 0.1       | 90.67 | 78.78 | 68.34 |
| Main Recurrent Dropout 0.5 -> 0.6 | 85.23 | 78.59 | 71.83 |
| Both Recurrent Dropout 0.5 -> 0.6 | 84.66 | 79.00 | 72.81 |
| Main LSTM Size 50 -> 100          | 90.49 | 79.84 | 70.29 |
| Main LSTM Size 50 -> 40           | 83.93 | 78.12 | 71.40 |
| Char embedding 10 -> 20           | 85.53 | 79.10 | 71.41 |
| Batch size 32 -> 16               | 85.82 | 78.82 | 72.84 |
| LSTM -> GRU                       | 84.27 | 78.80 | 72.53 |
| Epochs 30 -> 60                   | 87.90 | 79.65 | 72.02 |

One can see that no substantial gain can be obtained by simple tuning.
There might be potential gains to be had by longer training. 
Also, careful LSTM size changes and regularization might also improve the
results, since the model is still slightly overfitting.

## Contributors
@vt257
@gorians
@sardaryannarek
@artkh24
@armantsh
@HaykoGasparyan
@Arman-Deghoyan
