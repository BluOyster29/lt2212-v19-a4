import gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import argparse
from nltk import word_tokenize, ngrams
from nltk.corpus import stopwords
import numpy as np
from sklearn.model_selection import train_test_split

#Part 1: Preprocessing

def fetchw2v():
    word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
    return word_vectors#, w2v_vocab

def retrieving_data(filename, language_name):
    '''
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
            line = [w for w in line if w in word_vectors]
            vocab.extend(line)
            language_text.append(line)
            if position == args.endline:
                break
    
    vocab = set(vocab)
    vocab = [w for w in vocab if w in word_vectors]
    print(vocab)

    return language_text, vocab

    print('Length of English lines: {}'.format(len(english_lines)))
    print('Length of French lines: {}'.format(len(fr_lines)))

def truncate_me(text1, text2):
    '''Evens out sentences of text1 and text2
    '''
    newlines1 = []
    newlines2 = [] 

    for sent1, sent2 in zip(text1,text2):
        i = min([len(sent1),len(sent2)])
        newlines1.append(sent1[:i])
        newlines2.append(sent2[:i])
    return newlines1, newlines2

# def gengrams(text):
#     trigrams = ngrams(text, 3, pad_left=True, pad_right=True, left_pad_symbol='<start>', right_pad_symbol='<end>')
#     return trigrams

def encode_onehot(vocab):
    '''Instructions:
    For the target (English) trigram language model p(t)
    your code needs to collect the vocabulary for both the training and test data
    and turn it into vectors

    Args: text: string
    Output: one_hot: dict of {word: vector}
    '''
    word_index = {j:i for i,j in enumerate(vocab)}
    one_hot = {w: np.zeros(len(vocab), dtype=int) for w in vocab}

    for word in list(one_hot): #iterate over vocabulary keys while it's changing
        i = word_index[word]
        one_hot[word][i] = 1 
    return one_hot

def construct_onehot(text, one_hot):
    '''Create one hot encoded ngrams:
    [hot+hot, class]
    '''
    print("Constructing trigram model.")
    grams = ngrams(text, 3, pad_left=True, pad_right=False, left_pad_symbol='<start>')

    onehot_vectors = []

    for gram in list(grams):
        label = gram[-1]
        vector = []
        for w in gram[:-1]:
            vector += list(one_hot[w])
        vector.append(label)
        onehot_vectors.append(vector)   
    return onehot_vectors

def sentencevectors(text):
    '''Generate Word2Vec vectors to feed into the NN.
    Args: text: list of list
    Output: vectors: dict: {word: w2v vector}
    '''
    vectors = {}
    for sentence in text:
        for word in sentence:
            if word == '<start>':
                vec = np.random.rand(1,300) #this isn't in here, so we need to add it
                vectors.update({word: vec})
            else:
                vec = word_vectors[word] #select the right vector for each word       
                vectors.update({word: vec})  
    return vectors

def split_data(s_lang, t_lang, T):
    '''Make a test/train split
    Args:
        s_lang, t_lang: list of sentences
        T: desired size of test data (e.g. 0.1, 0.4...)
    Output:
        s(ource)train, stest, t(arget)train, ttest
    '''

    strain, stest, ttrain, ttest = train_test_split(s_lang, t_lang, test_size=T) #should we shuffle? yes/no
    return strain, stest, ttrain, ttest

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
    args = parser.parse_args()

    word_vectors = fetchw2v()
    eng_lines, eng_vocab = retrieving_data('english_slice.txt', 'english')
    fr_lines, french_vocab = retrieving_data('french_slice.txt', 'french')
    #=====================================================================
    eng_lines, fr_lines = truncate_me(eng_lines,fr_lines)
    #=====================================================================
    s_lang_train, s_lang_test, t_lang_train, t_lang_test = split_data(eng_lines,fr_lines,args.test_range)
    # eng_w2v = sentencevectors(s_lang_train)
