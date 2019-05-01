import gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import argparse
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from sklearn.model_selection import train_test_split

#Part 1: Preprocessing

def fetchw2v():
    word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
    w2v_vocab = word_vectors.vocab()
    return word_vectors, w2v_vocab

def retrieving_data(filename, language_name, w2v_vocab):
    '''to do:
        - further tokenisation?

    Args:
        filename: .txt file of source/target language data
        language_name: string
    Output:
        language_text: list: tokenized data without stopwords
    '''
    vocab = ['<s>']
    language_text = []
    stop_words = stopwords.words(language_name)

    with open(filename, encoding ='UTF-8') as f:
        position = 0 # to keep track of reading position
        if args.startline:
            print('Starting at line {}.'.format(args.startline))
            for i in range(args.startline): # start at line "startline"
                position += 1
                line = f.readline()
        for line in f:
            position += 1
            line = word_tokenize(line) # tolower?
            vocab.extend(line)
            language_text.append(line)
            if position == args.endline:
                break
    
    vocab = set(vocab)
    #we need to skip vocabulary items that are not in word2vec
    vocab = [w for w in vocab if w in w2v_vocab]

    return language_text, vocab

    print('Length of English lines: {}'.format(len(english_lines)))
    print('Length of French lines: {}'.format(len(french_lines)))

def truncate_me(text1, text2):
    '''Zip languages together --Lin
    this returns a list of tuples that is the french/english line
    i have truncated it, i don't know if you think there is a better way to output 
    this bit? --Rob
    '''
    twin_lines = []
    for i in range(len(text1)):
        lens = [len(text1[i]), len(text2[i])]
        print(lens)
        twin_lines.append((text1[i][:min(lens)],text2[i][:min(lens)]))
        #print(len(twin_lines[i][0]), len(twin_lines[i][1])) #testing that it worked
    return twin_lines 

#Part 2: Vectorization

def gengrams(text):
    trigrams = ngrams(text, 3, pad_left=True, pad_right=False, left_pad_symbol='<s>')
    return trigrams

def onehot(text):
    '''Instructions:
    For the target (English) trigram language model p(t)
    your code needs to collect the vocabulary for both the training and test data
    and turn it into vectors

    Args: text: string
    Output: one_hot: dict of {word: vector}
    '''
    word_index = {j:i for i,j in enumerate(text)}
    one_hot = {w: np.zeros(len(text), dtype=int) for w in text}

    for word in list(one_hot): #iterate over vocabulary keys while it's changing
        i = word_index[word]
        one_hot[word][i] = 1 
    return one_hot

def sentencevectors(sentence):
    '''generate source lang w2v vectors
    '''
    vectors = []
    for word in sentence:
        if word == '<start>':
            vec = np.random.rand(1,300) #this isn't in here, so we need to add it
            vectors.append(vec)
        else:
            vec = word_vectors[word] #select the right vector for each word       
            vectors.append(vec)     
    return vectors

def split_data(data, T):
    '''To split our test and training. We could also do this manually if you prefer.
    Let's figure out what the data is supposed to look like first ;)
    Args:
        data: arrays
        T: desired size of test data (e.g. 0.1, 0.4...)
    Output:
        arrays
    '''

    X_train, X_test, y_train, y_test = train_test_split(data, test_size=T) #should we shuffle? yes/no

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert text to features")
    parser.add_argument("-S", "--start", metavar="S", dest="startline", type=int,
                    default=0,
                    help="What line of the input data file to start from. Default is 0, the first line.")
    parser.add_argument("-E", "--end", metavar="E", dest="endline",
                    type=int, default=100,
                    help="What line of the input data file to end on. Default is None, whatever the last line is.")
    parser.add_argument("-T", "--test", metavar="T", dest="test_range",
                    type=int, default=20, help="What percentage of the set will be test")
    parser.add_argument("-R", "--random", dest="random", action="store_true", default=False, help="Specify whether to get random training/test")
    parser.add_argument("-P", "--preprocessing", dest="prepro", action="store_true", default=False,
                        help="specifies whether or not to use preprocessing")
    args = parser.parse_args()

    word_vectors, w2v_vocab = fetchw2v()
    english_lines, eng_vocab = retrieving_data('english_slice.txt', 'english')
    french_lines, french_vocab = retrieving_data('french_slice.txt', 'french')
    truncate_me(english_lines,french_lines)
    lang_model_train = onehot(train)
    lang_model_test = onehot(test)