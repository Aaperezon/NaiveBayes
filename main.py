import string
from random import shuffle
from math import ceil, log
import re
import pandas as pd
# imdb_labelled

def normalize(Lines):
    normalizedLines = []
    for line in Lines:
        line = (''.join([i for i in line if i not in string.punctuation and i != '\t'])).strip().lower()
        normalizedLines.append([line[0:-1].strip(),int(line[-1])])
    return (normalizedLines)
    #for x in range(len(text)):
    #    print(f"{x} {text[x]} ")


def getTrainingAndEvaluationTexts(texts, trainingRate):
    shuffle(texts)
    trainingLen = ceil(trainingRate * len(texts))
    return texts[0:trainingLen], texts[trainingLen:]
    #regrasa dos arreglo 'trainingTexts' 'tests'



def getTableOcurrencesOfWords(texts):
    class_f = [0,0]
    counts = dict()
    for i in range(len(texts)):
        class_f[texts[i][1]] += 1
        words = texts[i][0].split()
        for word in words:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1
    frequency_per_word = []
    for word in counts:
        frequency_per_word.append( [word , getOcurrenceByClass(word, 0, texts), getOcurrenceByClass(word, 1, texts) ] )
    return frequency_per_word, {fw[0] : (fw[1], fw[2]) for fw in frequency_per_word},  class_f[0], class_f[1], log( (class_f[0]) / (f:= class_f[0]+class_f[1]) )  , log((class_f[1]) / f) 

def getOcurrenceByClass(attempted_string,attempted_class,texts):
    counter = 0
    for i in range(len(texts)):
        if(int(texts[i][1]) == attempted_class): 
            counter = counter + texts[i][0].split().count(attempted_string)
    return counter


    

def getTableOfProbabilities(texts, vocabulary_len,negative_words, positive_words ):
    probabilities=[["Word/Class", "PosN" , "PosP", "LogN", "LogP"]]

    for row_freq in texts:
        neg = log((int(row_freq[1]) + 1)/(int(negative_words) + vocabulary_len))
        pos = log((int(row_freq[2]) + 1)/(int(positive_words) + vocabulary_len))
        probabilities.append( [ row_freq[0] , row_freq[1] , row_freq[2],neg  , pos ] )
    return probabilities
file1 = open('imdb_labelled.txt', 'r')
Lines = file1.readlines()
normalized_data = normalize(Lines)



#normalized_data = [["hello i am here",1],["hello juan i am here",0],["bye i am here",1],["hello i am there",1]]
training_tests, training_texts  = getTrainingAndEvaluationTexts(normalized_data,0.9)

words_ocurrences, words_ocurrences_dict , negative_class_f, positive_class_f,  negative_class_p, positive_class_p = getTableOcurrencesOfWords(training_tests)

vocabulary_len = len(words_ocurrences)
row_total = len(training_tests)
print(f"vocabulary:{vocabulary_len} P(pos):{positive_class_f}/{row_total} P(neg):{negative_class_f}/{row_total}")

table_probabilities = getTableOfProbabilities(words_ocurrences,vocabulary_len,negative_class_f,positive_class_f)
print(f"positive:{positive_class_f}  negative:{negative_class_f}")

pd.DataFrame(data = table_probabilities).to_csv("imdb_labelled_model.csv", index = False, header=False)







def getTestTable(tests, frequency_class,negatives, positives,  v):
    test_table=[["Instance", "LogN" , "LogP", "Class", "Real Class"]]
    for test in tests:
        print(test)
        log_ = [{},{}]
        log_v = [negatives,positives]
        for word in test[0].split():
            if(word in log_[test[1]] ):
                log_[test[1] ][word] += 1
            else:
                log_[test[1] ][word] = 1
        for i,l in enumerate(log_):
            for k,f in l.items():
                print(f" freq: {(frequency_class[k][i] if(k in frequency_class ) else 0 ) } v:{v}")
                print(f" division: {(f+1)/((frequency_class[k][i] if(k in frequency_class ) else 0 ) + v )} ")
                print(f"n:{negatives} p:{positives}")
                log_v[i] += log( (f+1) / ( (frequency_class[k][i] if(k in frequency_class ) else 0 ) + v ) )
        test_table.append([test[0],log_v[0], log_v[1], int(log_v[1]>log_v[0]), test[1]])

    return test_table






get_test_table = getTestTable(training_texts, words_ocurrences_dict,negative_class_p, positive_class_p,vocabulary_len)

for i in get_test_table:
    print(i)



