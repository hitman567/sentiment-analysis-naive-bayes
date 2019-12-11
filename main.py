import re
import numpy as np
from read import classes,dataset,frequency_of_words,content,test_doc
from collections import defaultdict
#Total number of Documents
number_of_docs=sum(classes.values())
#print(number_of_docs)
#Calculating likelihood Probability p(w,c)
loglikelihood={}
loglikelihood = defaultdict(lambda:0,loglikelihood)
def probability(word,cat):
    if word not in dataset[cat] or word not in frequency_of_words:
        return 0
    return np.log(float((dataset[cat][word]+1)/(classes[cat]+number_of_docs)))

logprior=[0,0]

#Function for training the data
def naive_bayes():

    #Calculating logprior p(c)
    for i in dataset.keys():
        if(i=="1"):
            j=1
        else:
            j=0
        pc=float((classes[i]+1)/number_of_docs)
        logprior[j]=np.log(pc)

#    print("Logprior:", logprior)

    #pwc
    for line in content:
        for word in line:
#            print(word)
            loglikelihood[word]={"0":probability(word,"0"),"1":probability(word,"1")}

#    print("Loglikelihood:", loglikelihood)
#Predicted Target
pred_target=[]
target=[]
#Function for testing the data
def predict():
    sums={0:0,1:0}
    for i in dataset.keys():
        if(i=="1"):
            j=1
        else:
            j=0
        sums[j]=logprior[j]
        for k in test_doc:
            category1,target1,filename1,content1=k.split(sep=" ",maxsplit=3)
            content1=content1[:-2]
            if(target1=="neg"):
                target.append(0)
            else:
                target.append(1)
            content1=re.split('[^a-zA-Z\']',content1)
            for word in content1:
                prob0=1
                prob1=1
                if word!=word.upper() and word in frequency_of_words:
                    prob0*=probability(word,"0")
                    prob1*=probability(word,"1")
                    sums[j]=loglikelihood[word][i]
            if(prob0>prob1):
                pred_target.append(0)
            else:
                pred_target.append(1)
#    print(sums)
    return sums


naive_bayes()
predict()
#print(target)
#print(pred_target)

cnt=0

tp=0
tn=0
fp=0
fn=0
n=len(target)
for i in range(len(target)):
    if(target[i]==1 and pred_target[i]==1):
        tp+=1
for i in range(len(target)):
    if(target[i]==0 and pred_target[i]==1):
        fp+=1
for i in range(len(target)):
    if(target[i]==1 and pred_target[i]==0):
        fn+=1
for i in range(len(target)):
    if(target[i]==0 and pred_target[i]==0):
        tn+=1
Accuracy=(tp+tn)/n
Precision=tp/(tp+fp)
Recall=tp/(tp+fn)
print("Accuracy:", (tp+tn)/n*100, "%")
print("Precision:", tp/(tp+fp))
print("Recall:", tp/(tp+fn))
f1=2*(Precision*Recall)/(Precision+Recall)
print("F1:",f1)
