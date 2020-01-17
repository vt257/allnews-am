import sentence
import random
from transformers import AutoTokenizer

def partitionRankings(data, percent):
    howManyNumbers = int(round(percent*len(data)))
    shuffled = list(data[:])
    random.shuffle(shuffled)
    return shuffled[howManyNumbers:], shuffled[:howManyNumbers]

def writeInfile(data, filename):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    subword_len_counter = 0
    with open(filename, 'wt',encoding='utf-8') as f:
        for sentence in data:
            for (token, key) in sentence:
                current_subwords_len = len(tokenizer.tokenize(token))
                if current_subwords_len == 0:
                    continue
                if (subword_len_counter + current_subwords_len) > 128:
                    f.write("\n")
                    f.write((token+' '+key+'\n'))
                    subword_len_counter = 0
                    continue
                subword_len_counter += current_subwords_len
                f.write((token+' '+key+'\n'))
            f.write('\n')

if __name__ == '__main__':
    traindevgetter=sentence.Sentence('pioner-silver/train.conll03')
    testgetter = sentence.Sentence('pioner-silver/dev.conll03')

    train,dev=partitionRankings(traindevgetter.tagged_sentences, 0.1)

    writeInfile(list(train),'train.txt')
    writeInfile(list(dev), 'dev.txt')
    writeInfile(list(testgetter.tagged_sentences), 'test.txt')


