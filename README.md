# lt2212-v19-a4
Group project for assignment 4

## Suggested alternative group names 
* The tuples
* Led Runagain?
* Jeff Buggley
* Debug Mode 
* Bruce Stringsteen
* Prints
* Fleetwood Stack

# Preparation

## PyTorch 
The tutorial for the Data Parallelism is [here](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

## Data 
I have downloaded the data and they are massive files so it might be good to save them outside the git repo just to avoid accidentally trying to push them, they're like 200mbs compressed... 
The Google News vectors is also massive, like over a gb so you might want to save these locally and we will keep them out of the git repo. 

# Part 1:Preprocessing 

## To do:
* args for randomly selecting data-size
* Ideas for tokenisation (not required but might be nice to do) e.g lowercase, stopwords,regexes
* Lines must be truncated so they are same length e.g line 1: Je suis Rob, line 2: my name is not Rob, then line two will have to have the last word cut off so that they are the same length. It's going to be a really naive translator so the output will look kinda silly. Old school google translate

## Can put below who does what for part 1:
* Hemanthu:
  ...
  
* Linnea:
...

* Rob:
...

# Part 2: Vectorisation

## Instructions
We need to collect the vocabulary for both the training and test data then turn it into vectors. We will not use one-hot vectors for the input layer (need to work out what this is) but we will need them for the output. The target language will use a trigram language model p(t)

### Input:
The input will be pretrained word2vec 300-dimensional vectors that we will load using Python gensim (i don't know what this is). You can find that on mltgpu in /scratch/GoogleNews-vectors-negative300.bin.gz. I have not added this to the repo because it is 1.4gbs... So make sure not to put it in there cos when we push it will be like trying to flush a brick down a toilet. save it locally for testing i think. 

The vectors can be accessed using gensim's [KeyedVector](https://www.pydoc.io/pypi/gensim-3.2.0/autoapi/models/keyedvectors/index.html).

From Asad: 

*Aside from the input vectors not being one-hot, from an input and output perspective, LaTeX: p(t) p ( t )  will look just like assignment 3, but you will have implemented the "guts" of the model this time.* I am not 100 percent what this means, so we will have to talk about this. 

We will have to keep track of start symbols e.g _<start>, <end>_. We also need to ignore vocab items that are missing from word2vec, this means we will have to skip both the english and french word. e.g *if i not in vocab: skip*
  
From Asad:

*LaTeX: p(s|t) p ( s | t )  will look a little different. It will take word2vec vectors from English as input, and predict one-hot vectors in French. So you will need, in other words, word2vec English vectors for the inputs of both models, one-hot English vectors for predicting the last word of a trigram, and one-hot French vectors for predicting the current word translation.*

Not totally sure what this means but we can work this out together

# Part 3: Simple PyTorch FFNN from scratch 

# Part 4: Training and testing 

# Part 5: Reporting and submission

# Bonus Part A: GPU

# Bonus Part B: A different model

# Bonus Part C: Party
Post assignment party 


