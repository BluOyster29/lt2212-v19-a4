import gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import torch

word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

'''we need to remove words that are not in here from the corpus
'''
print('length of w2v vocab:', len(word_vectors.wv.vocab))
w2v_vocab = word_vector.vocab()

'''We could use SVD:s again (Singular vector decomposition)
to get more dense vectors?'''

# def sig(x):
#     return 1/(1+np.exp(-x))

def softmax(inputs):
    return np.exp(inputs)/float(sum(np.exp(inputs)))

# '''initial output:
# weighted sum of the first layers activation (+bias)?
# '''

pred_out = sig(weighted_sum)

# '''back propagation part'''

class NNetwork:
    '''According to Asad:
    input -> layer -> sigmoid -> layer -> softmax -> loss
    '''
    def __init__(self, lr=1.0): #I just set the lr to the same as in Asads perceptron. Idk :)
        self.lr=lr
        '''randomly initialize weights (and bias?)
        torch.mm gets the dot product'''
        self.W1 = torch.randn(len(input), len(output), requires_grad=True)
        # self.W2 = 
        self.b1 = torch.randn(len(input), len(output), requires_grad=True)
        # self.b2 = 

        # def forward(self)
        # '''feed forward part'''
        #     l1 = torch.mm(feature_set, weights) #weighted sum
        #     out = torch.sigmoid(l1+self.bias)            

        #     l2 = softmax()

        # def train(self, data, classes, epochs)
        #     for e in range(0,epochs):

        #         loss = ?
        #         loss.backwardS()

        self.optimizer = torch.optim.Adam([self.W1, self.b1, self.W2, self.b2], lr=self.learning_rate)

        '''reset the gradient'''
        self.optimizer.zero_grad()

# '''loss function?'''
# loss = sum([(output-actual_value)**2 for neuron in layer])