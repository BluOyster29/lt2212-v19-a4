import gensim
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from nltk import FreqDist

# for efficiency
def read_lines2(path,l1 = 0,l2 = 1):
    with open(path) as train_file:
        head = [next(train_file) for x in range(l2)]
    return head

# repeat something mtimes in a list
def mtimes(x,m):
    return [x for i in range(m)]

# ngrams :: [Str] -> Int -> [[Str,Str,Str]]
def ngrams(xs,m):
    xs = mtimes('<start>',m) + xs
    return [xs[(n-(m-1)):(n+1)] for n in range(len(xs))][(m):]

# foldr :: a -> b -> [a] -> a -> b
def foldr(f,xs,z):
    acc = z
    for x in xs:
        acc = f(acc,x)
    return acc

def cc(x,y):
    return x + y

# for one-hots
# dict_map2 :: Dict (String:Int) -> Dict (String:ndarray)
def dict_map2(ds):
    dictionaryKeys = [key for key in ds]
    l = len(dictionaryKeys)
    values = np.identity(l)
    return dict(zip(dictionaryKeys, values))

def equalize(e1,f1):
    xs = [x.split() for x in [e1,f1]]
    ys = [len(x) for x in xs]
    m = min(ys[0],ys[1])
    ef = [xs[i][:m] for i in range(2)]
    return ef

# note calls readlines2
def eqEngandFrench(epath,fpath,n1,n2):
    e = read_lines2(epath,n1,n2)
    f = read_lines2(fpath,n1,n2)
    return [equalize(*list(zip(e,f))[i]) for i in range(len(e)) ]

def trigramsToData(tgs):
    rv300 = np.random.uniform(-.25,.25,(1,300))[0]
    dtgs = []
    for sen in tgs:
        dsen = []
        for tg in sen:
            bg = tg[:2]
            onehot = engOneHots[tg[2]]
            dbg = []
            for word in bg:
                if word == '<start>':
                    a = rv300
                else:
                    a = word_vectors[word]
                dbg.append(a)
            dsen.append((np.concatenate((dbg[0],dbg[1])),onehot))
        dtgs.append(dsen)
    return dtgs

engPath = "UN-english.txt"
frePath = "UN-french.txt"
folder = 'GoogleNews-vectors-negative300.bin'

word_vectors = KeyedVectors.load_word2vec_format(folder, binary=True)

equalEngandFrenchSentences = eqEngandFrench(engPath,frePath,0,100000)

# [[(Str,Str)]] [Zipped Sentences]
zippedTrans = [list(zip(x[0],x[1])) for x in equalEngandFrenchSentences]
# [[(Str,Str)]] [Zipped Sentences excluding non word2Vec]
filteredZippedTransL = [[w for w in zt0 if w[0] in word_vectors.vocab] for zt0 in zippedTrans]

# for generation of a dicitonary
allPairs = foldr(cc,filteredZippedTransL,[])
engDict = FreqDist([x[0] for x in allPairs])
freDict = FreqDist([x[1] for x in allPairs])
engOneHots = dict_map2(engDict)
freOneHots = dict_map2(freDict)

# tranlsated data
wvsOHs = [[(word_vectors[w[0]],freOneHots[w[1]]) for w in f] for f in filteredZippedTransL]

filteredEng = [[x[0] for x in f] for f in filteredZippedTransL]
engTrigrams = [ngrams(x,3) for x in filteredEng]
engTgData = trigramsToData(engTrigrams)

len(wvsOHs)
len(wvsOHs[0])
len(wvsOHs[0][0])
len(wvsOHs[0][0][0])
len(wvsOHs[0][0][1])

len(engTgData)#sentences
len(engTgData[0])#words in sentence
len(engTgData[0][0])# 
len(engTgData[0][0][0])# 
len(engTgData[0][0][1])# 

