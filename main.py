#TCSS 456 - Final Project
#Amanda Aldrich
#Taylor Riccetti
#Zira Cook

import json
import os.path as path
import sys
import random
import string
import nltk
from nltk.corpus import stopwords
from nltk import classify
from collections import Counter

STOP_LIST = stopwords.words('english')
NOUN_TAGS = ['NN', 'NNS', 'NNP', 'NNPS']
VERB_TAGS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
FIRST_PERSON_PRONOUNS = ['i', 'me', 'my', 'mine']


#Gets the first 150 reviews with review text only, returned as an array
def getPlainReviewText(file):
	translator = str.maketrans('', '', string.punctuation)
	reviews = []
	for i in range(150):
		j = json.loads(file.readline())
		reviews.append(j['reviewText'].translate(translator))
	return reviews


#Prepares data to be evaluated
#Used to extract features in get_features()
def trainReview(reviewText):
    tokens = nltk.word_tokenize(reviewText)
    tagged = nltk.pos_tag(tokens)
    return tagged


#Returns a dictionary of each word and its POS tag
#Helper method used in getFeatures to check for noun and verb frequency
def trainReviewDict(reviewText):
	tokens = nltk.word_tokenize(reviewText)
	tagged = nltk.pos_tag(tokens)
	tagDict = {}
	for tag in tagged:
		tagDict[tag[0]] = tag[1]
	return tagDict


#Used in classifier to determine if a review was written by a bot
def getFeatures(text):
    #1st person pronouns ('I', 'me')
    #More verbs than nouns
    #Upper case, word.isupper() 
    
    features = {}
    taggedWords = trainReview(text)
    tagDict = trainReviewDict(text)

    words = text.split()
    for word in words:
    	if word not in STOP_LIST:
    		if word.isupper():
    			features[word] = True
    		elif word in taggedWords:
    			features[word] = True
    			if tagDict[word] in NOUN_TAGS:
    				features[word] = False
    		elif word.lower() in FIRST_PERSON_PRONOUNS:
    			features[word] = True
    return features


#Divides up data and tags it as 'bot' or 'human', based on manual tagging
def getManualTrainData(appReviews, accessoryReviews):
    botReviews = []
    humanReviews = []
    idx = 0
    with open("bot_train_apps.txt", "r") as appFile:
        for line in appFile:
            if 'f' in line:
                botReviews.append(appReviews[idx])
                idx += 1
            if 't' in line:
                humanReviews.append(appReviews[idx])
                idx += 1
    idx = 0
    with open("bot_train_accessories.txt", "r") as accessoryFile:
        for line in accessoryFile:
            if 'f' in line:
                botReviews.append(accessoryReviews[idx])
                idx += 1
            if 't' in line:
                humanReviews.append(accessoryReviews[idx])
                idx += 1

    return botReviews, humanReviews


#Combines all reviews into one large list, while retaining the 'bot' and 'human' tags
def getAllReviews(botReviews, humanReviews):
    bots = [(review, 'bot') for review in botReviews]
    humans = [(review, 'human') for review in humanReviews]
    allReviews = bots + humans
    return allReviews


def printClassifierEval(trainSet, testSet, classifier):
	print ('Accuracy on the training set = ' + str(classify.accuracy(classifier, trainSet)))
	print ('Accuracy of the test set = ' + str(classify.accuracy(classifier, testSet)))


def main():
    dirUpTwo = path.abspath(path.join(__file__, "../.."))
    apps = open(dirUpTwo + "/reviews_Apps_for_Android_5.json", "r")
    accessories = open(dirUpTwo + "/reviews_Cell_Phones_and_Accessories_5.json", "r")

    appReviews = getPlainReviewText(apps)
    accessoryReview = getPlainReviewText(accessories)
    botReviews, humanReviews = getManualTrainData(appReviews, accessoryReview)

    allReviews = getAllReviews(botReviews, humanReviews)
    random.shuffle(allReviews)
    
    featureSet = [(getFeatures(review), tag) for (review, tag) in allReviews]
    setlen = int(len(featureSet) / 2)
    trainSet, testSet = featureSet[setlen:], featureSet[:setlen]
    classifier = nltk.NaiveBayesClassifier.train(trainSet)

    printClassifierEval(trainSet, testSet, classifier)
    print(classifier.classify(getFeatures("Easy to use.")))
    
    apps.close()
    accessories.close()

main()