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
The processed corpus can be found 
[here [~800MB]](https://storage.googleapis.com/allnews_am/corpus_1679k.zip). 
This is produces by the `generate_corpus.py` script.
Download it, unzip it and put the `corpus_1679k` file it in the `data` folder. 
After this, you should be able to run a word2vec training using e.g.
```
python scripts/train_word2vec.py --epochs=5 --corpus=allnews_am/data/corpus_1679k
```
which will save the model files in the `models` folder after
the training is complete. Similarly, you can train FastText using
```
python scripts/train_fasttext.py --epochs=5  --corpus=allnews_am/data/corpus_1679k
```
You can also run
```
python scripts/train_word2vec.py --help
```
for more training options.

### Evaluation
You can run a basic evaluation using
```
python scripts/explore_embeddings.py
```
Some results on the `coarse_avetisyan_ghukasyan_analogies.txt` are
shown below, for models trained on the 100k news corpus:

| Model | Semantic | Syntactic | Total |
| --- | --- | --- | --- |
| fastText_sg_100_lr0025_ep30 |  4.0% (74/1858) |  50.0% (2537/5069) | 37.7% (2611/6927) |
| word2vec_sg_100_lr0025_ep30 |  9.1% (169/1858) |  4.8% (241/5069) |  5.9% (410/6927) |

For the 1679k news corpus:

| Model | Semantic | Syntactic | Total |
| --- | --- | --- | --- |
| fastText_sg_200_lr01_ep10 |  0.5% (17/3392) |  46.8% (3346/7149) | 31.9% (3363/10541) |
| fastText_sg_100_lr0025_ep5 |  2.8% (95/3392) |  45.6% (3260/7149) |  31.8% (3355/10541) |

Both models were trained using the default hyperparameters.

**NOTE** We are not lowercasing words and we keep the punctuation,
so the scores on the analogy tasks are expected to be worse in comparison
to text pre-processing that performs these steps.

## Contributors
@vt257
@gorians
@sardaryannarek
