# allnews-am
Natural Language Processing for Armenian News Summarization.

## Requirements
- Set up environment variables `ALLNEWS_AM_MYSQL_HOST`, `ALLNEWS_AM_MYSQL_NAME`, `ALLNEWS_AM_MYSQL_USER`, 
`ALLNEWS_AM_MYSQL_PASS` for database access.

## Setup
1. Run `pip install -r requirements.txt`
1. Run `pip install .`

## Word2vec 
### Training
The processed Text8 corpus can be found 
[here](https://storage.googleapis.com/allnews_am/corpus). 
This is produces by the `generate_corpus.py` script.
Download it and put it in the `data` folder. After 
this you should be able to run a training using e.g.
```
python scripts/train_word2vec.py --epochs=5
```
which will save the model files in the `models` folder after
the training is complete. Run
```
python scripts/train_word2vec.py --help
```
for more training options.

### Evaluation
You can run a basic evaluation using
```python
python scripts/explore_word2vec.py
```
g
**NOTE** We are not doing lowercasing words and we keep the punctuation,
so the scores on the analogy tasks are expected to be worse in comparison
to text pre-processing that performs these steps.

## Contributors
@vt257
@gorians
