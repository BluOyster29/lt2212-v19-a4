import os, sys, glob, argparse, numpy as np, pandas as pd, subprocess, re
from nltk import word_tokenize

#Part 1: Preprocessing

def process_args():

    ''' Here we will add the parsers to the program so we can save them as 
    variables'''

    parser = argparse.ArgumentParser(description="Convert text to features")
    parser.add_argument("-S", "--start", metavar="S", dest="startline", type=int,
                    default=0,
                    help="What line of the input data file to start from. Default is 0, the first line.")
    parser.add_argument("-E", "--end", metavar="E", dest="endline",
                    type=int, default=100,
                    help="What line of the input data file to end on. Default is None, whatever the last line is.")
    parser.add_argument("-T", "--test", metavar="T", dest="test_range",
                    type=int, default=20, help="What percentage of the set will be test")
    parser.add_argument("-R", "--random", metavar="R", dest="random", action="store_true", default=False, help="Specify whether to get random training/test")
    parser.add_argument("-P", "--preprocessing", metavar="P", dest="prepro", action="store_true", default=False,
                        help="specifies whether or not to use preprocessing")
    args = parser.parse_args()
    
    return args.startline, args.endline, args.test_range, args.random, args.prepro

def retrieving_data(english, french):

    ''' to do:
            - start/endline functionality
            - percentage of set to be used training/test
            - further tokenisation
            - truncation 
    '''

    french_lines = []
    with open(english, encoding ='UTF-8') as en:
        
        #room here for further tokenisation, we can discuss what we should use
        english_lines = [word_tokenize(i) for i in en]
        '''
        for i in en:
            #line = i.split(' ')
            line = word_tokenize(i)
            lines.append(line)
        '''
        en.close()
    with open(french, encoding="UTF-8") as fr:
        french_lines = [word_tokenize(i) for i in fr]
        fr.close()

    twin_lines = []
    for i in range(len(english_lines)):
        lens = [len(english_lines[i]), len(french_lines[i])]
        print(lens)
        twin_lines.append((english_lines[i][:min(lens)],french_lines[i][:min(lens)]))
        #print(len(twin_lines[i][0]), len(twin_lines[i][1])) #testing that it worked

    return twin_lines '''this returns a list of tuples that is the french/english line
                        i have truncated it, i don't know if you think there is a better way to output 
                        this bit?'''
                          
#Part 2: Vectorisation


if __name__ == '__main__':
    #retrieving_data('french_slice.txt')
    retrieving_data('english_slice.txt', 'french_slice.txt')