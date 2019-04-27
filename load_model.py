import gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import torch

word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

'''we need to remove words that are not in here from the corpus
'''
print('length of w2v vocab:', len(word_vectors.wv.vocab))
w2v_vocab = word_vector.vocab()

# Print this to see if model works
test = word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])
print(test)

'''We could use SVD:s again (Singular vector decomposition)
to get more dense vectors!'''

def sig(x):
    return 1/(1+np.exp(-x))

def softmax(inputs):
    return np.exp(inputs)/float(sum(np.exp(inputs)))

# '''initial output:
# weighted sum of the first layers activation (+bias)?
# '''

pred_out = sig(weighted_sum)

# '''back propagation part'''

class NNetwork:
    def __init__(self, lr=1.0):
        self.lr=lr
        '''randomly initialize weights (and bias?)
        torch.mm gets the dot product'''
        W = torch.rand(len(input), len(output))
        #b = ?

        # '''feed forward part
        # torch.mm signifies dot product'''

        w_sum = torch.mm(feature_set, weights) + bias #weighted sum

        # hidden layer 1 <<< Is there really just 1 hidden layer?
        # hidden layer 2

# '''loss function'''
# loss = sum([(output-actual_value)**2 for neuron in layer])