
# Given: French

def return2():
    return 1,2

a,b = return2()

def translate_sent(french_sent):

for frech_word_e in enumerate(french_sent):
    translations = []
    if frech_word_e[0] == 0: 
        w1,w2 = initWordProb(frech_word_e[1],'<s>','<s>')
        translations.append(w2)
    else:
        w1,w2 = initWordProb(frech_word_e[1],w1,w2)
        translations.append(w2)
    return translations

def initWordProb(fw,w1,w2):
    if w1 == '<s>':
        englishVocab = fullEngVocab
    else:
        englishVocab = topN(n,)
    ehatList = []
    for e in englishVocab:
        p_e_given_w1w2 = trigModel(w1,w2)
        pe = p_e_given_startstart[e] #e.g. the probability of that word
        p_f_given_e = transModel(e)
        pf = p_f_given_e[fw]
        ehat.append((e,pf*pe))
    return w2, maxOfList(ehat)

def topN(n):
    w1 = prevprevWord
    w2 = prevWord
    p_e_given_w1_w2 = trigModel(w1,w2)
    return projectTopNIndexs(p_e_given_w1_w2,n)

# so we should be able to separate trigram and transaltion model data


def initWordProb():
    ehatList = []
    for e in englishVocab:
        p_e_given_startstart = trigModel(start,start)
        pe = p_e_given_startstart[e] #e.g. the probability of that word
        p_f_given_e = transModel(e)
        pf = p_f_given_e[f0]
        ehat.append((e,pf*pe))
    return maxOfList(ehat)


# can realistically just call the above function where the trigram model acts differently based off input parameters


def restWordProb(n):
    ehatList = []
    w1 = prevprevWord
    w2 = prevWord
    for e in englishVocab:
        p_e_given_start_w = trigModel(w1,w2)
        pe = p_e_given_startstart[e] #e.g. the probability of that word
        p_f_given_e = transModel(e)
        pf = p_f_given_e[f0]
        ehat.append((e,pf*pe))
    return maxOfList(ehat)







