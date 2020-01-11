import os
import gensim.models

file_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(file_dir, "../allnews_am/models/word2vec.model")
model = gensim.models.Word2Vec.load(model_dir)

print(f'Vocabulary size: {len(model.wv.vocab)}')
print('Most common words: ', [
      (model.wv.index2word[i], model.wv.vocab[model.wv.index2word[i]].count)
      for i in range(30)
])
similar_words_to = 'Փաշինյան'
print(f'Most similar to "{similar_words_to}"',
      model.wv.most_similar(similar_words_to))
