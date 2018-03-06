#TCSS 456 - Final Project
#Amanda Aldrich
#Taylor Riccetti
#Zira Cook
import json
import os.path as path
import sys

import nltk

def train(file):
	# Read in the first 150 lines of each category and train.
	i = 1
	for i in range(150):
		j = json.loads(file.readline())
		text = nltk.word_tokenize(j['reviewText'])
		tagged = nltk.pos_tag(text)
		print(tagged)


def main():
	dirUpTwo = path.abspath(path.join(__file__, "../.."))
	apps = open(dirUpTwo + "//Apps_for_Android_5.json", "r")
	accessories = open(dirUpTwo + "//Cell_Phones_and_Accessories_5.json", "r")

	train(apps)
	train(accessories)

	apps.close()
	accessories.close()

main()

