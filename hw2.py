import nltk
import math
import scipy
import pickle
import random
import numpy as np
import pandas as pd
import os,sys,re,csv
from numba import jit
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cosine
from collections import Counter, defaultdict

# Instead of fullrec being a list of onehots, make it a list of numbers representing the index of where the one-hot should be one, list comprehension

#... (1) First load in the data source and tokenize into one-hot vectors.
#... Since one-hot vectors are 0 everywhere except for one index, we only need to know that index.


#... (2) Prepare a negative sampling distribution table to draw negative samples from.
#... Consistent with the original word2vec paper, this distribution should be exponentiated.


#... (3) Run a training function for a number of epochs to learn the weights of the hidden layer.
#... This training will occur through backpropagation from the context words down to the source word.

#... (4) Test your model. Compare cosine similarities between learned word vectors.

#.................................................................................
#... global variables
#.................................................................................

random.seed(10)
np.random.seed(10)
randcounter = 10
np_randcounter = 10

vocab_size = 0
hidden_size = 100
uniqueWords = [""]                      #... list of all unique tokens
wordcodes = {}                          #... dictionary mapping of words to indices in uniqueWords
wordcounts = Counter()                  #... how many times each token occurs
samplingTable = []                      #... table to draw negative samples from

#.................................................................................
#... load in the data and convert tokens to one-hot indices
#.................................................................................
def loadData(filename):
    global uniqueWords, wordcodes, wordcounts, fullrec, fullrec_filtered
    override = True
    if override:
        #... for debugging purposes, reloading input file and tokenizing is quite slow
        #...  >> simply reload the completed objects. Instantaneous.
        fullrec = pickle.load(open("w2v_fullrec.p","rb"))
        wordcodes = pickle.load( open("w2v_wordcodes.p","rb"))
        uniqueWords= pickle.load(open("w2v_uniqueWords.p","rb"))
        wordcounts = pickle.load(open("w2v_wordcounts.p","rb"))
        return fullrec

    # ... load in first 15,000 rows of unlabeled data file.  You can load in
    # more if you want later (and should do this for the final homework)
    handle = open(filename, "r", encoding="utf8")
    fullconts = handle.read().split("\n")
    # fullconts = fullconts[1:15000]
    # (TASK) Use all the data for the final submission

    #... apply simple tokenization (whitespace and lowercase)
    fullconts = [" ".join(fullconts).lower()]

    print ("Generating token stream...")
    #... (TASK) populate fullrec as one-dimension array of all tokens in the order they appear.
    #... ignore stopwords in this process
    #... for simplicity, you may use nltk.word_tokenize() to split fullconts.
    #... keep track of the frequency counts of tokens in origcounts.

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(fullconts[0])

    fullrec = []
    for word in word_tokens:
        if word not in stop_words:
            fullrec.append(word)

    min_count = 50
    origcounts = Counter(fullrec)

    print ("Performing minimum thresholding..")
    #... (TASK) populate array fullrec_filtered to include terms as-is that appeared at least min_count times
    #... replace other terms with <UNK> token.
    #... update frequency count of each token in dict wordcounts where: wordcounts[token] = freq(token)
    fullrec_filtered = []
    for x in fullrec:
        if origcounts[x] > min_count:
            fullrec_filtered.append(x)
        else:
            # if x[:-3] == 'ing':
            #     fullrec_filtered.append("UNKing")
            # if x[:-2] == 'ly':
            #     fullrec_filtered.append("UNKly")
            # if x[:-2] == 'ed':
            #     fullrec_filtered.append("UNKed")
            # if x[:-2] == 'er':
            #     fullrec_filtered.append("UNKer")
            # else:
            fullrec_filtered.append("UNK")

    wordcounts = Counter(fullrec_filtered)

    #... after filling in fullrec_filtered, replace the original fullrec with this one.
    fullrec = fullrec_filtered

    print ("Producing one-hot indicies")
#     #... (TASK) sort the unique tokens into array uniqueWords
#     #... produce their one-hot indices in dict wordcodes where wordcodes[token] = onehot_index(token)
#     #... replace all word tokens in fullrec with their corresponding one-hot indices.
    uniqueWords = []
    for word in wordcounts:
        uniqueWords.append(word)

    wordcodes = {}

    i = 0
    for word in uniqueWords:
        zeros = [0] * len(uniqueWords)
        zeros[i] = 1
        wordcodes[word] = zeros
        i+=1

    i = 0
    for word in fullrec:
        fullrec[i] =  uniqueWords.index(word)
        i+=1

    # one_hots = []
    # wordcodes_inverse = {}
    # for value in wordcodes.values():
    #     one_hots.append(value)

    #... close input file handle
    handle.close()

#     #... store these objects for later.
#     #... for debugging, don't keep re-tokenizing same data in same way.
#     #... just reload the already-processed input data with pickles.
#     #... NOTE: you have to reload data from scratch if you change the min_count, tokenization or number of input rows

    pickle.dump(fullrec, open("w2v_fullrec.p","wb+"))
    pickle.dump(wordcodes, open("w2v_wordcodes.p","wb+"))
    pickle.dump(uniqueWords, open("w2v_uniqueWords.p","wb+"))
    pickle.dump(dict(wordcounts), open("w2v_wordcounts.p","wb+"))

    #... output fullrec should be sequence of tokens, each represented as their one-hot index from wordcodes.
    return fullrec

# uniqueWords is a list of all the eunique words
# wordcodes is a dictionary with with the unique words as keys and the one hot encodings of the words as the values
# wordcounts is a dictionary with the unique words as keys and counts of those words as values
# fullrec is a list of the one hot encoding of words in the order that they appeared

#.................................................................................
#... compute sigmoid value
#.................................................................................

@jit
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

#.................................................................................
#... generate a table of cumulative distribution of words
#.................................................................................

def negativeSampleTable(train_data, uniqueWords, wordcounts, exp_power=0.75):
    global cumulative_dict

    total_words = []
    for x in uniqueWords:
        total_words.append(wordcounts[x])
    total_words = sum(total_words)

    # global wordcounts
    #... stores the normalizing denominator (count of all tokens, each count raised to exp_power)
    max_exp_count = 0

    print("Generating exponentiated count vectors")
    #... (TASK) for each uniqueWord, compute the frequency of that word to the power of exp_power
    #... store results in exp_count_array.

    exp_count_array = []
    for word in uniqueWords:
        exp_count_array.append((wordcounts[word]/total_words)**exp_power)

    max_exp_count = sum(exp_count_array)

    print ("Generating distribution")

    #... (TASK) compute the normalized probabilities of each term.
    #... using exp_count_array, normalize each value by the total value max_exp_count so that
    #... they all add up to 1. Store this corresponding array in prob_dist
    prob_dist = []
    i = 0
    for x in exp_count_array:
        normal = x/max_exp_count
        prob_dist.append(normal)
        i+=1

    print ("Filling up sampling table")
    #... (TASK) create a dict of size table_size where each key is a sequential number and its value is a one-hot index
    #... the number of sequential keys containing the same one-hot index should be proportional to its prob_dist value
    #... multiplied by table_size. This table should be stored in cumulative_dict.
    #... we do this for much faster lookup later on when sampling from this table.

    global cumulative_dict
    cumulative_dict = {}

    table_size = 1e7

    i = 0
    w = 0
    for prob in prob_dist:
        num_times = prob*table_size
        for x in range(int(num_times)):
            word = uniqueWords[w]
            code = wordcodes[word]
            cumulative_dict[i]=word
            i+=1
        w+=1

    return cumulative_dict

#.................................................................................
#... generate a specific number of negative samples
#.................................................................................

def generateSamples(context, num_samples):
    #... (TASK) randomly sample num_samples token indices from samplingTable.
    #... don't allow the chosen token to be context_idx.
    #... append the chosen indices to results
    global samplingTable, uniqueWords, randcounter, wordcodes
    length = len(cumulative_dict)
    # change this so the one hot encoding isnt picked -- right now we just have it so the index isnt picked

    results = []

    context = uniqueWords[context]

    for num in range(num_samples):
        random_index = random.randrange(0, length)
        while cumulative_dict[random_index] == wordcodes[context]:
            random_index = random.randrange(0, length)
        results.append(random_index)

    return results

@jit(nopython=True)
def performDescent(target_W2_index, hidden, lr, W1, W2, context_W2_index, neg1_context_W2_index, neg2_context_W2_index):

    nll_new = 0

    E = -(math.log(sigmoid(np.dot(W2[context_W2_index], hidden))))
    E_neg = (math.log(sigmoid(np.dot(W2[neg1_context_W2_index], hidden)))) + (math.log(sigmoid(np.dot(W2[neg2_context_W2_index], hidden))))
    nll_new = E - E_neg

    v_j = W2[context_W2_index]
    summation = lr * (sigmoid(np.dot(np.transpose(v_j), hidden))-1) * v_j
    W2[context_W2_index] = v_j - lr * (sigmoid(np.dot(np.transpose(v_j), hidden))-1) * hidden

    v_j = W2[neg1_context_W2_index]
    summation = summation + (lr * sigmoid(np.dot(np.transpose(v_j), hidden))-0) * v_j
    W2[neg1_context_W2_index] = v_j - lr * (sigmoid(np.dot(np.transpose(v_j), hidden))-0) * hidden

    v_j = W2[neg2_context_W2_index]
    summation = summation + (lr * sigmoid(np.dot(np.transpose(v_j), hidden))-0) * v_j
    W2[neg2_context_W2_index] = v_j - lr * (sigmoid(np.dot(np.transpose(v_j), hidden))-0) * hidden

    v_i = W1[target_W2_index]
    W1[target_W2_index] = W1[target_W2_index] - lr * summation

    return nll_new

#.................................................................................
#... learn the weights for the input-hidden and hidden-output matrices
#.................................................................................
from tqdm import tqdm

def trainer(curW1 = None, curW2=None):
    global uniqueWords, wordcodes, fullsequence, vocab_size, hidden_size, np_randcounter, randcounter, nll_results
    vocab_size = len(uniqueWords)           #... unique characters
    hidden_size = 100                       #... number of hidden neurons
    context_window = [-2,-1,1,2]            #... specifies which context indices are output. Indices relative to target word. Don't include index 0 itself.
    nll_results = []                        #... keep array of negative log-likelihood after every 1000 iterations

    start_point = int(math.fabs(min(context_window)))
    end_point = len(fullsequence)-(max(max(context_window),0))
    mapped_sequence = fullsequence

    if curW1==None:
        np_randcounter += 1
        W1 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
        W2 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
    else:
        W1 = curW1
        W2 = curW2
    epochs = 30
    num_samples = 2
    learning_rate = 0.05
    nll = 0
    iternum = 0

    unique_words_one_hots = []
    for value in wordcodes.values():
        unique_words_one_hots.append(value)

    t = len(fullsequence)

    for j in range(0,epochs):
        print ("Epoch: ", j)
        prevmark = 0
        #... For each epoch, redo the whole sequence...

        with tqdm(total=t) as pbar:

            for i in range(start_point,end_point): #here since our window is 2, the start point is the third word and the endpoint is the third to last word

                # if (float(i)/len(mapped_sequence))>=(prevmark+0.1):
                #     print ("Progress: ", round(prevmark+0.1,1))
                #     prevmark += 0.1
                if iternum%10000==0:
                    # print ("Negative likelihood: ", nll)
                    nll_results.append(nll)
                    nll = 0

                center_token = mapped_sequence[i]

                # #... (TASK) don't allow the center_token to be <UNK>. move to next iteration if you found <UNK>.
                if center_token == 25:
                    continue

                hidden = W1[center_token]

                iternum += 1

                for k in context_window:
                     #... (TASK) Use context_window to find one-hot index of the current context token.
                    context_index = mapped_sequence[i + k]

                    # One iteration is going to go through one context word and 2 negative examples
                     #... construct some negative samples for descent
                    negative_indices = generateSamples(context_index, num_samples)

                    context_matrix_index = mapped_sequence[context_index]

                    neg1_context_W2_index = cumulative_indexes[negative_indices[0]]

                    neg2_context_W2_index = cumulative_indexes[negative_indices[1]]

                    nll = performDescent(center_token, hidden, learning_rate, W1, W2, context_matrix_index, neg1_context_W2_index, neg2_context_W2_index)

                pbar.update(1)

    for nll_res in nll_results:
        print (nll_res)
    return [W1,W2]

#.................................................................................
#... Load in a previously-saved model. Loaded model's hidden and vocab size must match current model.
#.................................................................................

def load_model():
    handle = open("saved_W1.data","rb")
    W1 = np.load(handle)
    handle.close()
    handle = open("saved_W2.data","rb")
    W2 = np.load(handle)
    handle.close()
    return [W1,W2]

#.................................................................................
#... Save the current results to an output file. Useful when computation is taking a long time.
#.................................................................................

def save_model(W1,W2):
    handle = open("saved_W1.data","wb+")
    np.save(handle, W1, allow_pickle=False)
    handle.close()

    handle = open("saved_W2.data","wb+")
    np.save(handle, W2, allow_pickle=False)
    handle.close()

#... so in the word2vec network, there are actually TWO weight matrices that we are keeping track of. One of them represents the embedding
#... of a one-hot vector to a hidden layer lower-dimensional embedding. The second represents the reversal: the weights that help an embedded
#... vector predict similarity to a context word.

#.................................................................................
#... code to start up the training function.
#.................................................................................
word_embeddings = []
proj_embeddings = []
def train_vectors(preload=False):
    global word_embeddings, proj_embeddings
    if preload:
        [curW1, curW2] = load_model()
        # [word_embeddings, proj_embeddings] = load_model()
    else:
        curW1 = None
        curW2 = None
    [word_embeddings, proj_embeddings] = trainer(curW1,curW2)
    save_model(word_embeddings, proj_embeddings)

#.................................................................................
#... for the averaged morphological vector combo, estimate the new form of the target word
#.................................................................................

# def morphology(word_seq):
#     global word_embeddings, proj_embeddings, uniqueWords, wordcodes
#     embeddings = word_embeddings
#     vectors = [word_seq[0], # suffix averaged
#     embeddings[wordcodes[word_seq[1]]]]
#     vector_math = vectors[0]+vectors[1]
#     #... find whichever vector is closest to vector_math
#     #... (TASK) Use the same approach you used in function prediction() to construct a list
#     #... of top 10 most similar words to vector_math. Return this list.
#
# #.................................................................................
#... for the triplet (A,B,C) find D such that the analogy A is to B as C is to D is most likely
#.................................................................................

def analogy(word_seq):
    global word_embeddings, proj_embeddings, uniqueWords, wordcodes
    embeddings = word_embeddings
    vectors = [embeddings[wordcodes[word_seq[0]]],
    embeddings[wordcodes[word_seq[1]]],
    embeddings[wordcodes[word_seq[2]]]]
    vector_math = -vectors[0] + vectors[1] - vectors[2] # + vectors[3] = 0

    word_1 = word_embeddings[uniqueWords.index(word_seq[0])]
    word_2 = word_embeddings[uniqueWords.index(word_seq[1])]
    word_3 = word_embeddings[uniqueWords.index(word_seq[2])]

    vector_math = -word_1 + word_2 - word_3

    # target_simm = 1 - scipy.spatial.distance.cosine(word_1, word_2)
    #
    # simmilarities = []
    # i=0
    # for x in word_embeddings:
    #     simm = 1 - scipy.spatial.distance.cosine(word_3, x)
    #     # simmilarities.append(abs(target_simm - simm))
    #     simmilarities.append(tuple((uniqueWords[i], (abs(target_simm - simm)))))
    #     i+=1
    #
    # sorted_similarity = sorted(simmilarities, key=lambda tup: tup[1])
    # sorted_similarity = list(reversed(sorted_similarity))
    # top_10 = sorted_similarity[0:9]
    #
    # print('\n')
    # for x in top_10:
    #     print(x)
    # print('\n')
    # return top_10


    # calculate similarity between first two words and put them in a list
    # loop through word_embeddings and find the similarity and out them in a list
    # subtract one list from another and the number closest to 0 will be the answer

    simmilarities = []
    i = 0
    for x in word_embeddings:
        simm = 1 - scipy.spatial.distance.cosine(x, vector_math)
        simmilarities.append(tuple((uniqueWords[i], simm)))
        i+=1
    # simmilarities.sort(reverse = True)
    sorted_similarity = sorted(simmilarities, key=lambda tup: tup[1])
    sorted_similarity = list(reversed(sorted_similarity))
    top_10 = sorted_similarity[1:10]

    print('\n')
    for x in top_10:
        print(x)
    print('\n')
    return top_10

    # print(vector_math)
    # print(len(vector_math))
    #... find whichever vector is closest to vector_math
    #... (TASK) Use the same approach you used in function prediction() to construct a list
    #... of top 10 most similar words to vector_math. Return this list.

    #.................................................................................
#... find top 10 most similar words to a target word
#.................................................................................

def get_neighbors(target_word):
    global word_embeddings, uniqueWords, wordcodes
    targets = [target_word]
    outputs = []
    similarities = []

    target_word_embedding = word_embeddings[uniqueWords.index(target_word)]

    i = 0

    for word in uniqueWords:
        embedding = word_embeddings[uniqueWords.index(word)]
        similarity = 1 - scipy.spatial.distance.cosine(target_word_embedding, embedding)
        similarities.append(tuple((word, similarity)))

    sorted_similarity = sorted(similarities, key=lambda tup: tup[1])
    sorted_similarity = list(reversed(sorted_similarity))
    top_10 = sorted_similarity[1:10]

    with open('/Users/img/Desktop/630/Week4/hw2/prob7_output.txt', 'a') as f:
        f.write("Top 10 words most similar to "+target_word+"\n")

    print('\n')
    print("top 10 words most similar to "+target_word)
    for x in top_10:
        print(x)
        with open('/Users/img/Desktop/630/Week4/hw2/prob7_output.txt', 'a') as f:
            f.write(str(x)+'\n')
    print('\n')
    with open('/Users/img/Desktop/630/Week4/hw2/prob7_output.txt', 'a') as f:
        f.write('\n\n')
    return top_10

    # ... (TASK) search through all uniqueWords and for each token, compute its similarity to target_word.
    # ... you will compute this using the absolute cosine similarity of the word_embeddings for the word pairs.
    # ... Note that the cosine() function from scipy.spatial.distance computes a DISTANCE -- (1 - distance) --so you need to convert that to a similarity.
    # ... return a list of top 10 most similar words in the form of dicts,
    # ... each dict having format: {"word":<token_name>, "score":<cosine_similarity>}

fullsequence = loadData('unlabeled-data.txt')
print ("Full sequence loaded...")
print(wordcounts)
print ("Total unique words: ", len(uniqueWords))
print ("Total words: ", len(fullrec))

print("Preparing negative sampling table")
samplingTable = negativeSampleTable(fullsequence, uniqueWords, wordcounts)

# Un-comment when training
# cumulative_indexes = list(cumulative_dict.values())
# i = 0
# with tqdm(total=len(cumulative_indexes)) as pbar:
#     for x in cumulative_indexes:
#         cumulative_indexes[i] = uniqueWords.index(x)
#         i+=1
#         pbar.update(1)



# print(uniqueWords.index("UNK"))
# train_vectors(preload=False)
[word_embeddings, proj_embeddings] = load_model()
# print(word_embeddings)
# get_neighbors('carbohydrates')
# get_neighbors('california')

analogy(['red', 'apple', 'yellow'])



# Un-comment to test intrinsic data
# intrinsic = pd.read_csv('intrinsic-test.csv')
# print(intrinsic.head(15))
# word1 = list(intrinsic['word1'])
# word2 = list(intrinsic['word2'])
# id = list(intrinsic['Id'])
#
#
# cosine_similarity = []
# i=0
# for word in word1:
#     word1_embedding = word_embeddings[uniqueWords.index(word)]
#     word2_embedding = word_embeddings[uniqueWords.index(word2[i])]
#     cos_sim = 1 - scipy.spatial.distance.cosine(word1_embedding, word2_embedding)
#     cosine_similarity.append(cos_sim)
#     i+=1
#
#
# output = pd.DataFrame(columns=['id', 'sim'])
# output['id'] = id
# output['sim'] = cosine_similarity
#
# output.to_csv("/Users/img/Desktop/630/Week4/hw2/cosine_outputs.csv", index=False)




# if __name__ == '__main__':
#     if len(sys.argv)==2:
#         filename = sys.argv[1]
#         # feel free to raed the file in a different way
#         #... load in the file, tokenize it and assign each token an index.
#         #... the full sequence of characters is encoded in terms of their one-hot positions
#
#         fullsequence = loadData(filename)
#         print ("Full sequence loaded...")
#         # print(uniqueWords)
#         # print (len(fullrec))
#
#
#
#
#         #... now generate the negative sampling table
#         print ("Total unique words: ", len(uniqueWords))
#         print ("Total words: ", len(fullrec))
#         print("Preparing negative sampling table")
#         samplingTable = negativeSampleTable(fullsequence, uniqueWords, wordcounts)
#
#
#         # ... we've got the word indices and the sampling table. Begin the training.
#         # ... NOTE: If you have already trained a model earlier, preload the results (set preload=True) (This would save you a lot of unnecessary time)
#         # ... If you just want to load an earlier model and NOT perform further training, comment out the train_vectors() line
#         # ... ... and uncomment the load_model() line
#
#         train_vectors(preload=False)
#
#
#
#
#
#
#
#         # print(len(word_embeddings))
#         # one = word_embeddings[uniqueWords.index('honey')]
#         # get_neighbors('awesome')
#         # word_embeddings[uniqueWords.index(target_word)]
#         # two = word_embeddings[uniqueWords.index('cheese')]
#         # three = word_embeddings[uniqueWords.index('blood')]
#         # similarity = 1 - scipy.spatial.distance.cosine(one, two)
#         # similarity1 = 1 - scipy.spatial.distance.cosine(one, three)
#         # print(similarity)
#         # print(similarity1)
#
#         #
#         # ... we've got the trained weight matrices. Now we can do some predictions
#         # ...pick ten words you choose
#         # targets = ["good", "bad", "food", "apple",'tasteful','unbelievably','uncle','tool','think']
#         # for targ in targets:
#         #     print("Target: ", targ)
#         #     bestpreds= (prediction(targ))
#         #     for pred in bestpreds:
#         #         print (pred["word"],":",pred["score"])
#         #     print ("\n")
#
#
#         #
#         # #... try an analogy task. The array should have three entries, A,B,C of the format: A is to B as C is to ?
#         # print (analogy(["apple", "fruit", "banana"]))
#         #
#         # #... try morphological task. Input is averages of vector combinations that use some morphological change.
#         # #... see how well it predicts the expected target word when using word_embeddings vs proj_embeddings in
#         # #... the morphology() function.
#         # #... this is the optional task, if you don't want to finish it, common lines from 545 to 556
#         #
#         # s_suffix = [word_embeddings[wordcodes["banana"]] - word_embeddings[wordcodes["bananas"]]]
#         # others = [["apples", "apple"],["values", "value"]]
#         # for rec in others:
#         #     s_suffix.append(word_embeddings[wordcodes[rec[0]]] - word_embeddings[wordcodes[rec[1]]])
#         # s_suffix = np.mean(s_suffix, axis=0)
#         # print (morphology([s_suffix, "apples"]))
#         # print (morphology([s_suffix, "pears"]))
#
#     else:
#         print ("Please provide a valid input filename")
#         sys.exit()
