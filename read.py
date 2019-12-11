import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import random

f=open("naive_bayes_data.txt","r")

text=f.readlines()

#Splitting the data
random.shuffle(text)
train_doc = text[:int((len(text)+1)*.80)] #Remaining 80% to training set
test_doc = text[int(len(text)*.80+1):] #Splits 20% data to test set

text=train_doc
target=[]
filename=[]
content=[]
classes={"1":0,"0":0}
dataset={"1":{},"0":{}}
frequency_of_words={}

#Creating the dictionary for storing the frequency of words.
for i in text:
    category1,target1,filename1,content1=i.split(sep=" ",maxsplit=3)
    temp="0"
    if(target1=="neg"):
        classes["0"]+=1
        target.append(0)
    else:
        temp="1"
        classes["1"]+=1
        target.append(1)

    filename.append(filename1)
    content1=content1[:-2]
    content1=re.split('[^a-zA-Z\']',content1)
    for word in content1:
        if(word!=word.upper()):
            dataset[temp].setdefault(word,0)
            dataset[temp][word]+=1
            frequency_of_words.setdefault(word,{})
            frequency_of_words[word].setdefault(temp,0)
            frequency_of_words[word][temp]+=1
#     print(content1)
    content.append(content1)
# print(target)
# print(filename)
# print(content)
print(frequency_of_words)

