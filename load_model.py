import gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import torch
import random

parser = argparse.ArgumentParser(description="Convert text to features")
parser.add_argument("-I", "--epochs", metavar="I", dest="iterations", type=int, default=1,
                    help="Numbers of desired epochs.")

args = parser.parse_args()


word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

'''we need to remove words that are not in here from the corpus
'''
print('length of w2v vocab:', len(word_vectors.wv.vocab))
w2v_vocab = word_vector.vocab()

# def sig(x):
#     return 1/(1+np.exp(-x))

# def softmax(inputs):
#     return np.exp(inputs)/float(sum(np.exp(inputs)))

class NNetwork:
    '''According to Asad:
    input -> layer -> sigmoid -> layer -> softmax -> loss
    '''
    def __init__(self, lr=0.1): #I just set the lr to the same as in Asads perceptron. Idk :)
        self.lr=lr

        def forward(self, features)
        '''feed forward part'''
            l1 = features.mm(self.W1)
            out = torch.sigmoid(l1+self.b1)          
            l2 = out.mm(self.W2)
            out2 = torch.sigmoid(l2+self.b2)

            # Warrick's one liner
            # out = features.mm(self.W1).sigmoid().mm(self.W2)

        def train(self, data, classes, epochs=args.iterations)
            '''What corresponds to what in the equation in the instructions:
            W1 = W
            W2 = U
            b1 = b
            b2 = c


            To do:
            Make a minibatch!
            '''

            self.W1 = torch.randn(len(features), len(output), requires_grad=True)
            self.W2 = torch.randn(len(output), len(nextlayer), requires_grad=True)
            self.b1 = torch.randn(1, requires_grad=True)
            self.b2 = torch.randn(1, requires_grad=True)

            '''back propagation part'''
            self.optimizer = torch.optim.Adam([self.W1, self.b1, self.W2, self.b2], lr=self.lr)

            for e in range(0,epochs):
                for i in range(batchsize):
                    print('Epoch number {}'.format(e))
                    self.forward()
                    loss = (pred - actual).pow(2).sum()
                    self.optimizer.zero_grad()
                    loss.backwards()
                    self.optimizer.step()

        def predict(self, features):
            '''Again, like Asads perceptron... but not sure if this is
            the right place to use the softmax.
            '''
            predictions = []
            for i in range (0, len(features)):
                d = features.mm(self.W1)
                out = torch.sigmoid(d+self.b1) 
                d = out.mm(self.W2)
                predicted = torch.softmax(d+self.b2)
                self.output = predicted
            return predictions

        
'''loss function?
loss = sum([(output-actual_value)**2 for neuron in layer])
loss = (y_pred - y).pow(2).sum()'''