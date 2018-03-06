#TCSS 456 - Final Project
#Amanda Aldrich
#Taylor Riccetti
#Zira Cook
import json
import os.path as path
import sys

import nltk

def main():
	dirUpTwo = path.abspath(path.join(__file__, "../.."))
	apps = open(dirUpTwo + "//Apps_for_Android_5.json", "r")
	accessories = open(dirUpTwo + "//Cell_Phones_and_Accessories_5.json", "r")


	#Read in the first 150 lines of each category and train.
	i = 1
	for i in range(150):
		appJson = json.loads(apps.readline())
		print(appJson['reviewText'])

		accessJson = json.loads(accessories.readline())
		print(accessJson['reviewText'])

	apps.close()
	accessories.close()

main()