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
    text = text.translate(TRANSLATOR)
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
    print('Accuracy on the training set = %.2f' % classify.accuracy(classifier, trainSet))
    print('Accuracy of the test set = %.2f' % classify.accuracy(classifier, testSet))
    print('\n')


# Reads all reviews from a given file
# Returns all review text as a list
def getTestReviews(file):
    print("Retrieving all reviews from ", file.name)
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
        bot = 0.01
        human = 0.01
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
    progressCount = 0
    for review in reviews:
        progressCount += 1
        progress(progressCount, 10000)
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
    totalPronouns = 0.01
    firstPersonPronouns = 0.01
    for word in review:
        if word[1] == "PRP":
            totalPronouns += 1
            lword = word[0].lower()
            if lword in FIRST_PERSON_PRONOUNS:
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


# Classifies all given reviews
# Returns what percentage is bot and what percentage is human
def classifiyData(classifier, reviews):
    print("\nClassifying review data")
    totalReviews = 0
    botReviews = 0
    humanReviews = 0
    for inputReview in reviews:
        totalReviews += 1
        progress(totalReviews, 10000)
        result = classifier.classify(getFeatures(inputReview))
        if result is "bot":
            botReviews += 1
        else:
            humanReviews += 1

    return (botReviews / totalReviews)*100, (humanReviews / totalReviews)*100


# A progress bar for the bulk data classifier
def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()


# Tests all data for bots with the classifier
def testClassifierOnData(appTestReviews, accessoriesTestReviews, classifier):
    appBotPercent, appHumanPercent = classifiyData(classifier, appTestReviews)
    print("\n", appBotPercent,"% of app reviews are bots, and ", appHumanPercent,"% of app reviews are human")

    accessoryBotPercent, accessoryHumanPercent = classifiyData(classifier, accessoriesTestReviews)
    print("\n", accessoryBotPercent,"% of accessory reviews are bots and", accessoryHumanPercent,"% of accessory reviews are human")



def main():
    print("Welcome to Group 1's Amazon Review Analyzer.\nPlease wait while we train our classifier...")
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

    appTestReviews = getTestReviews(apps)
    accessoriesTestReviews = getTestReviews(accessories)

    print("\nTesting classifier...")
    testClassifierOnData(appTestReviews, accessoriesTestReviews, classifier)

    print("Testing frequencies...")

    print("We are testing frequencies of our words and their tags now...\nA bot review often has a different grammatical makeup and "
          + "collection of the parts of speech.\nFor example, you may see more first person pronouns in a bot written review.")

    botTagFreq = getFrequencies(botReviews)
    humanTagFreq = getFrequencies(humanReviews)

    start = time.time()
    print("\nTesting apps...")
    botAverageApps, humanAverageApps = frequenciesOfAll(botTagFreq, humanTagFreq, appTestReviews)
    print("\n", botAverageApps,"% of app reviews are bots, and", humanAverageApps,"% are human")
    
    print("\nTesting accessories...")
    botAverageAccessories, humanAverageAccessories = frequenciesOfAll(botTagFreq, humanTagFreq, accessoriesTestReviews)
    print("\n", botAverageAccessories,"% of accessory reviews are bots, and", humanAverageAccessories,"% are human")

    print("Train complete.\n")
    end = time.time()
    elapsedTime = (end - start) / 60
    print("Elapsed time (minutes): ", elapsedTime)
    manualDataCrawlBotCheck(apps, accessories)

    print("Training complete.\n")

    # Testing a string, with punctuation removed
    inputReview = input("Please enter a review to test against our data: ")
    inputProduct = input("Please enter the type of product being reviewed (press enter to skip this part of the analysis): ")
    while (inputReview is not "\n" and inputReview is not "" and (
            inputReview is not "exit" or inputProduct is not "exit")):
        inputReview = inputReview.translate(TRANSLATOR)
        inputProduct = inputProduct.translate(TRANSLATOR)
        print("Running your input review through the classifier...")
        classResult = classifier.classify(getFeatures(inputReview))

        print("Calculating frequencies of your review...")
        freqBotResult, freqHumanResult = compareFrequencies(botTagFreq, humanTagFreq, inputReview.translate(TRANSLATOR))

        if (inputProduct is not ""):
            print("Calculating words similar to product type and listing a few on screen...")
            similarWords = perecentageSimilarToProduct(inputReview, inputProduct)

        print("\nCalculating pronoun frequency...")
        proFreq = pronounFrequency(inputReview)

        print("Calculating verb to noun ratios...")
        nounFreq, verbFreq = verbsToNouns(inputReview)


        print("\nThe classifier believes your review is a " + classResult + "\n")
        print("Compared to bot tag frequencies this review had %.2f %% similar frequencies" % freqBotResult)
        print("Compared to human tag frequencies this review had %.2f %% similar frequencies" % freqHumanResult)
        if (freqBotResult > freqHumanResult):
            print("Given the frequency of tags, this review is believed to be a bot written review.\n")
        else:
            print("Given the frequency of tags, this review is believed to be a human written review.\n")

        if (inputProduct is not ""):
            print("%.2f %% of words in the review are similar to the product type.\n" % similarWords)

        print("%.2f %% pronouns in our review.\nA higher percentage of pronouns in a review can indicate a bot wrote the review\n" % proFreq)
        
        print("%.2f %% nouns in our review.\nA higher percentage of nouns in a review as compared to verbs can indicate a human wrote the review.\n" % nounFreq)

        print("%.2f %% verbs in our review\nA higher percentage of verbs in a review as compared to nouns can indicate a bot wrote the review.\n" % verbFreq)

        inputReview = input("Please enter a review to test: ")
        inputProduct = input("Please enter the product of the review (press enter to skip this part of the analysis): ")
    apps.close()
    accessories.close()


main()
