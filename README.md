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
The processed Text8 corpus can be found 
[here](https://storage.googleapis.com/allnews_am/corpus). 
This is produces by the `generate_corpus.py` script.
Download it and put it in the `data` folder. After 
this you should be able to run a word2vec training using e.g.
```
python scripts/train_word2vec.py --epochs=5
```
which will save the model files in the `models` folder after
the training is complete. Similarly, you can train FastText using
```
python scripts/train_fasttext.py --epochs=5
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
shown below:

| Model | Semantic | Syntactic | Total |
| --- | --- | --- | --- |
| fastText_sg_100 |  4.0% (74/1858) |  50.0% (2537/5069) | 37.7% (2611/6927) |
| word2vec_sg_100 |  9.1% (169/1858) |  4.8% (241/5069) |  5.9% (410/6927) |

Both models were trained using the default hyperparameters.

**NOTE** We are not lowercasing words and we keep the punctuation,
so the scores on the analogy tasks are expected to be worse in comparison
to text pre-processing that performs these steps.

## Contributors
@vt257
@gorians
@sardaryannarek
