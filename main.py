# TCSS 456 - Final Project
# Amanda Aldrich
# Taylor Riccetti
# Zira Cook

import json
import os.path as path
import sys
import random
import string
import time
import nltk
from nltk.corpus import stopwords
from nltk import classify
from collections import Counter

TRANSLATOR = str.maketrans('', '', string.punctuation)
STOP_LIST = stopwords.words('english')
NOUN_TAGS = ['NN', 'NNS', 'NNP', 'NNPS']
VERB_TAGS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
FIRST_PERSON_PRONOUNS = ['i', 'me', 'my', 'mine']


# Review object, contains all variables for JSON fields
class Review:
    def __init__(self, asin, reviewerId, helpful, reviewText, score, title, unixReviewTime, reviewTime):
        self.asin = asin
        self.reviewerId = reviewerId
        self.helpful = helpful
        self.reviewText = reviewText
        self.score = score
        self.title = title
        self.unixReviewTime = unixReviewTime
        self.reviewTime = reviewTime
        self.isBot = False


# Reads in JSON from both review files and creates a dict of them based on product number (asin)
def createReviewObjectsDict(file):
    reviews = {}
    file.seek(0)
    for line in file:
        j = json.loads(line)
        asin = j['asin']
        reviewObject = Review(asin, j['reviewerID'], j['helpful'], j['reviewText'], j['overall'], j['summary'], j['unixReviewTime'], j['reviewTime'])
        if asin in reviews:
            reviews[asin].append(reviewObject)
        else:
            reviews[asin] = []
            reviews[asin].append(reviewObject)

    return reviews



# Gets the first 150 reviews with review text only, returned as an array
def getPlainReviewText(file):
    reviews = []
    for i in range(150):
        j = json.loads(file.readline())
        reviews.append(j['reviewText'].translate(TRANSLATOR))  # All punctuation removed
    return reviews


# Prepares data to be evaluated
# Used to extract features in get_features()
def trainReview(reviewText):
    tokens = nltk.word_tokenize(reviewText)
    tagged = nltk.pos_tag(tokens)
    return tagged


# Returns a dictionary of each word and its POS tag
# Helper method used in getFeatures to check for noun and verb frequency
def trainReviewDict(reviewText):
    tokens = nltk.word_tokenize(reviewText)
    tagged = nltk.pos_tag(tokens)
    tagDict = {}
    for tag in tagged:
        tagDict[tag[0]] = tag[1]
    return tagDict


# Used in classifier to determine if a review was written by a bot
def getFeatures(text):
    # 1st person pronouns ('I', 'me')
    # More verbs than nouns
    # Upper case, word.isupper()

    features = {}
    taggedWords = trainReview(text)
    tagDict = trainReviewDict(text)

    words = text.split()
    for word in words:
        if word not in STOP_LIST:
            if word.isupper():
                features[word] = False
            elif word in taggedWords:
                features[word] = True
                if tagDict[word] in VERB_TAGS:
                    features[word] = False
            elif word.lower() in FIRST_PERSON_PRONOUNS:
                features[word] = False
    return features


# Divides up data and tags it as 'bot' or 'human', based on manual tagging
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


# Combines all reviews into one large list, while retaining the 'bot' and 'human' tags
def getAllReviews(botReviews, humanReviews):
    bots = [(review, 'bot') for review in botReviews]
    humans = [(review, 'human') for review in humanReviews]
    allReviews = bots + humans
    return allReviews


# Prints the accuracy of the classifier
def printClassifierEval(trainSet, testSet, classifier):
    print('Accuracy on the training set = ' + str(classify.accuracy(classifier, trainSet)))
    print('Accuracy of the test set = ' + str(classify.accuracy(classifier, testSet)))


def getTestReviews(file):
    file.seek(0)
    reviews = []
    for line in file:
        j = json.loads(line)
        reviews.append(j['reviewText'].translate(TRANSLATOR))  # All punctuation removed
    return reviews


# Finds POS tag frequency for a set of review texts
def getFrequencies(reviews):
    reviewFreq = {}
    for bot in reviews:
        tokens = nltk.word_tokenize(bot)
        tagged = nltk.pos_tag(tokens)
        for (word, tag) in tagged:
            if tag in reviewFreq:
                reviewFreq[tag] = reviewFreq[tag] + 1
            else:
                reviewFreq[tag] = 1
    return reviewFreq


# Finds POS tag frequency for a given review text
def getSingleReviewFrequency(review):
    reviewFreq = {}
    tagged = nltk.pos_tag(nltk.word_tokenize(review))
    for (word, tag) in tagged:
        if tag in reviewFreq:
            reviewFreq[tag] = reviewFreq[tag] + 1
        else:
            reviewFreq[tag] = 1
    return reviewFreq


# Compares the frequencies of test data with frequencies of input review.
def compareFrequencies(botFreq, humanFreq, inputReview):
    inputTagFreq = getSingleReviewFrequency(inputReview)
    totalBotFreq = 0.01
    reviewFreq = 0.01
    totalHumanFreq = 0.01
    for tag in inputTagFreq:
        bot = 0
        human = 0
        if tag in botFreq:
            bot += botFreq[tag]
        if tag in humanFreq:
            human += humanFreq[tag]

        totalBotFreq += bot
        totalHumanFreq += human
        reviewFreq += inputTagFreq[tag]

    return (reviewFreq / totalBotFreq) * 100, (reviewFreq / totalHumanFreq) * 100


def frequenciesOfAll(botFreq, humanFreq, reviews):
    totalBotFreq = 0
    totalHumanFreq = 0
    total = 0
    for review in reviews:
        botResult, humanResult = compareFrequencies(botFreq, humanFreq, review)
        total += botResult + humanResult
        totalBotFreq += botResult
        totalHumanFreq += humanResult
    return (totalBotFreq / total) * 100, (totalHumanFreq / total) * 100


# Calculates how similar words in the review are to the product.
# This can gauge how detailed the review is.
def perecentageSimilarToProduct(review, product):
    count = 0
    total = 0
    review = nltk.word_tokenize(review)
    product = nltk.word_tokenize(product)
    text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())

    for word in product:
        total += 1
        if (word in text):
            count += 1

        similiarWords = text.similar(word)
        if (similiarWords is not None):
            total += len(similiarWords)
            for similar in similiarWords:
                if (similar in review):
                    count += 1
    return count / total * 100


# If you're anxious about coming across as sincere, apparently you talk about yourself more.
# That's probably why words like 'I' and 'me' appear more often in fake reviews.
def pronounFrequency(review):
    review = nltk.pos_tag(nltk.word_tokenize(review))
    # I me my
    totalPronouns = 0.01
    firstPersonPronouns = 0.01
    for word in review:
        if word[1] == "PRP":
            totalPronouns += 1
            lword = word[0].lower()
            if (lword == "me" or lword == "i" or lword == "my"):
                firstPersonPronouns += 1
    return firstPersonPronouns / totalPronouns * 100


# Fakes tend to include more verbs as their writers often substitute pleasant (or alarming)
# sounding stories for actual insight. Genuine reviews are heavier on nouns.
def verbsToNouns(review):
    review = nltk.pos_tag(nltk.word_tokenize(review))

    totalNouns = 0.01
    totalNounsAndVerbs = 0.01
    totalVerbs = 0.01
    for word in review:
        if word[1] in NOUN_TAGS:
            totalNouns += 1
            totalNounsAndVerbs += 1
        if word[1] in VERB_TAGS:
            totalVerbs += 1
            totalNounsAndVerbs += 1

    return totalNouns / totalNounsAndVerbs * 100, totalVerbs / totalNounsAndVerbs * 100


# Scans all product reviews for duplicates
def checkForDuplicateReviewText(reviewDict):
    reviewTexts = {}
    textCounts = {}
    for product in reviewDict:
        reviews = reviewDict[product] # Gets the list of reviews associated with a product id

        for review in reviews:
            text = review.reviewText
            if text in reviewTexts:
                review.isBot = True
                textCounts[text] += 1
            else:
                reviewTexts[text] = review
                textCounts[text] = 1


# Holds method calls to data crawl bot checks
def manualDataCrawlBotCheck(appFile, accessoryFile):
    reviewDict = createReviewObjectsDict(appFile)
    reviewDict.update(createReviewObjectsDict(accessoryFile))
    checkForDuplicateReviewText(reviewDict)


def main():
    print("Training classifier...")
    dirUpTwo = path.abspath(path.join(__file__, "../.."))
    apps = open("reviews_Apps_for_Android.json", "r")
    accessories = open("reviews_Cell_Phones_and_Accessories.json", "r")

    # Parses all reviews and creates a clean list of all reviews
    appReviews = getPlainReviewText(apps)
    accessoryReview = getPlainReviewText(accessories)
    botReviews, humanReviews = getManualTrainData(appReviews, accessoryReview)
    allReviews = getAllReviews(botReviews, humanReviews)
    random.shuffle(allReviews)

    # Create feature set and classifier to test data
    featureSet = [(getFeatures(review), tag) for (review, tag) in allReviews]
    setlen = int(len(featureSet) / 2)
    trainSet, testSet = featureSet[setlen:], featureSet[:setlen]
    classifier = nltk.NaiveBayesClassifier.train(trainSet)
    printClassifierEval(trainSet, testSet, classifier)

    print("Testing frequencies...")

    botTagFreq = getFrequencies(botReviews)
    humanTagFreq = getFrequencies(humanReviews)

    manualDataCrawlBotCheck(apps, accessories)

    start = time.time()
    appTestReviews = getTestReviews(apps)
    print("Testing apps...")  										# This is not working.
    botAverageApps, humanAverageApps = frequenciesOfAll(botTagFreq, humanTagFreq, appTestReviews)
    print(str(botAverageApps) + "% of apps reviews are bots, and " + str(humanAverageApps) + "% are human.")
    end = time.time()
    elapsedTime = (end - start) / 60
    #print("Elapsed time (minutes): ", elapsedTime)
    print("Testing accessories...")
    accessoriesTestReviews = getTestReviews(accessories)
    botAverageAccessories, humanAverageAccessories = frequenciesOfAll(botTagFreq, humanTagFreq, accessoriesTestReviews)
    print(str(botAverageAccessories) + "% of accessory reviews are bots, and " + str(humanAverageAccessories) + "% are human.")
    print("Train complete.\n")

    # Testing a string, with punctuation removed
    inputReview = input("Type a review to test: ")
    inputProduct = input("Type the product of the review (press enter to skip this part of the analysis): ")
    while (inputReview is not "\n" and inputReview is not "" and (
            inputReview is not "exit" or inputProduct is not "exit")):
        inputReview = inputReview.translate(TRANSLATOR)
        inputProduct = inputProduct.translate(TRANSLATOR)
        print("Calculating classifier...")
        classResult = classifier.classify(getFeatures(inputReview))

        print("Calculating frequencies...")
        freqBotResult, freqHumanResult = compareFrequencies(botTagFreq, humanTagFreq, inputReview.translate(TRANSLATOR))

        if (inputProduct is not ""):
            print("Calculating words similar to product...")
            similarWords = perecentageSimilarToProduct(inputReview, inputProduct)

        print("Calculating pronoun frequency...")
        proFreq = pronounFrequency(inputReview)

        print("Calculating verb to noun ratios...")
        nounFreq, verbFreq = verbsToNouns(inputReview)

        print("The classifier belives this review is a " + classResult + "\n")
        print("Compared to bot tag frequencies this review had " + str(freqBotResult) + "% similar frequencies")
        print("Compared to human tag frequencies this review had " + str(freqHumanResult) + "% similar frequencies")
        if (freqBotResult > freqHumanResult):
            print("Given the frequency of tags, this review is believed to be a bot.\n")
        else:
            print("Given the frequency of tags, this review is believed to be a Human.\n")

        if (inputProduct is not ""):
            print(str(similarWords) + "% of words in the review are similar to the product.\n")

        print(str(proFreq) + "% a higher percent of pronouns appear more frequently in a fake review.\n")

        print(str(nounFreq) + "% a higher percentage of nouns shows a more genuine review.\n")

        print(str(verbFreq) + "% a higher percentage of verbs shows a less substance in a review.\n")

        inputReview = input("Type a review to test: ")
        inputProduct = input("Type the product of the review (press enter to skip this part of the analysis): ")
    apps.close()
    accessories.close()


main()
