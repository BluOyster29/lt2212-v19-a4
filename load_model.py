import torch
import random
import argparse

parser = argparse.ArgumentParser(description="Configure neural network")
parser.add_argument("-I", "--epochs", metavar="I", dest="epochs", type=int, default=1,
                    help="Numbers of desired epochs.")

args = parser.parse_args()

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

        def forward(self, features):
            '''feed forward part'''
            l1 = features.mm(self.W1)
            out = torch.sigmoid(l1+self.b1)          
            l2 = out.mm(self.W2)
            self.output = torch.softmax(l2+self.b2)
            # out2 = torch.sigmoid(l2+self.b2)

            # Warrick's one liner
            # out = features.mm(self.W1).sigmoid().mm(self.W2)

        def train(self, features, classes, epochs=args.iterations):
            '''What corresponds to what in the equation in the instructions:
            W1 = W
            W2 = U
            b1 = b
            b2 = c


            To do:
            Make a minibatch!

            features: X
            classes: Y
            '''

            self.W1 = torch.randn(len(features), len(output), requires_grad=True)
            self.W2 = torch.randn(len(output), len(nextlayer), requires_grad=True)
            self.b1 = torch.randn(1, requires_grad=True)
            self.b2 = torch.randn(1, requires_grad=True)

            '''back propagation part'''
            self.optimizer = torch.optim.Adam([self.W1, self.b1, self.W2, self.b2], lr=self.lr)

            for e in range(0,args.epochs):
                print('Epoch number {}'.format(e))
                for i in range(batchsize):
                    self.forward()
                    loss = (predicted - self.output).pow(2).sum()
                    self.optimizer.zero_grad()
                    loss.backwards()
                    self.optimizer.step()

        # def predict(self, features):
        #     '''Again, like Asads perceptron... but not sure if this is
        #     the right place to use the softmax.
        #     '''
        #     predictions = []
        #     for i in range (0, len(features)):
        #         d = features.mm(self.W1)
        #         out = torch.sigmoid(d+self.b1) 
        #         d = out.mm(self.W2)
        #         predicted = torch.softmax(d+self.b2)
        #         self.output = predicted
        #     return predictions
        
'''loss function?
loss = sum([(output-actual_value)**2 for neuron in layer])
loss = (y_pred - y).pow(2).sum()'''