from nltk.corpus.reader import ConllCorpusReader
from nltk.corpus.reader import LazyConcatenation, LazyMap, map_tag

class UpdatedConllReader(ConllCorpusReader):

    def _get_tagged_words(self, grid, tagset=None):
        pos_tags = self._get_column(grid, self._colmap['ne'])
        if tagset and tagset != self._tagset:
            pos_tags = [map_tag(self._tagset, tagset, t) for t in pos_tags]
        return list(zip(self._get_column(grid, self._colmap['words']), pos_tags))

class Sentence(object):
    def __init__(self, datapath):
        self.n_sent = 0
        corpusReader = UpdatedConllReader(datapath.split('/')[0], datapath.split('/')[1], ['words','pos','chunk', 'ne'])
        self.tagged_sentences=corpusReader.tagged_sents()
        self.sentences = [" ".join([s[0] for s in sent]) for sent in self.tagged_sentences]
        self.labels = [[s[1] for s in sent] for sent in self.tagged_sentences]
        self.tags_vals = list(set([num for elem in self.labels for num in elem]))
        self.tag2idx = {t: i for i, t in enumerate(self.tags_vals)}

    def get_next(self):
        try:
            s = self.sentences[self.n_sent]
            self.n_sent += 1
            return s
        except:
            return None

"""
getter = Sentence('pioner-silver/dev.conll03')
print(getter.tagged_sentences[0])
print(getter.sentences[0])
print(getter.labels[0])
print(getter.tags_vals)
"""