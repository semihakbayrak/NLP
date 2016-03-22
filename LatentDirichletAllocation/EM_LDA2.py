import numpy as np
import scipy as sc
from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import math
import random

stop = stopwords.words('english')
tokenizer = RegexpTokenizer(r'\w+')

doc1 = brown.words(fileids=['ch01']);
doc2 = brown.words(fileids=['cm01']);
doc3 = brown.words(fileids=['ch02']);
doc4 = brown.words(fileids=['cm02']);
doc5 = brown.words(fileids=['ch03']);
doc6 = brown.words(fileids=['ch04']);
doc7 = brown.words(fileids=['ch05']);
doc8 = brown.words(fileids=['ch06']);
doc9 = brown.words(fileids=['cm04']);
doc10 = brown.words(fileids=['ch07']);
doc11 = brown.words(fileids=['ch08']);
doc12 = brown.words(fileids=['cm05']);
doc13 = brown.words(fileids=['ch09']);
doc14 = brown.words(fileids=['cm06']);
doc15 = brown.words(fileids=['ch10']);

#text preprocessing
doc1_new = [j for j in doc1 if j not in stop]
doc1_last = tokenizer.tokenize((' '.join(doc1_new)).lower())
doc1 = doc1_last
doc2_new = [j for j in doc2 if j not in stop]
doc2_last = tokenizer.tokenize((' '.join(doc2_new)).lower())
doc2 = doc2_last
doc3_new = [j for j in doc3 if j not in stop]
doc3_last = tokenizer.tokenize((' '.join(doc3_new)).lower())
doc3 = doc3_last
doc4_new = [j for j in doc4 if j not in stop]
doc4_last = tokenizer.tokenize((' '.join(doc4_new)).lower())
doc4 = doc4_last
doc5_new = [j for j in doc5 if j not in stop]
doc5_last = tokenizer.tokenize((' '.join(doc5_new)).lower())
doc5 = doc5_last
doc6_new = [j for j in doc6 if j not in stop]
doc6_last = tokenizer.tokenize((' '.join(doc6_new)).lower())
doc6 = doc6_last
doc7_new = [j for j in doc7 if j not in stop]
doc7_last = tokenizer.tokenize((' '.join(doc7_new)).lower())
doc7 = doc7_last
doc8_new = [j for j in doc8 if j not in stop]
doc8_last = tokenizer.tokenize((' '.join(doc8_new)).lower())
doc8 = doc8_last
doc9_new = [j for j in doc9 if j not in stop]
doc9_last = tokenizer.tokenize((' '.join(doc9_new)).lower())
doc9 = doc9_last
doc10_new = [j for j in doc10 if j not in stop]
doc10_last = tokenizer.tokenize((' '.join(doc10_new)).lower())
doc10 = doc10_last
doc11_new = [j for j in doc11 if j not in stop]
doc11_last = tokenizer.tokenize((' '.join(doc11_new)).lower())
doc11 = doc11_last
doc12_new = [j for j in doc12 if j not in stop]
doc12_last = tokenizer.tokenize((' '.join(doc12_new)).lower())
doc12 = doc12_last
doc13_new = [j for j in doc13 if j not in stop]
doc13_last = tokenizer.tokenize((' '.join(doc13_new)).lower())
doc13 = doc13_last
doc14_new = [j for j in doc14 if j not in stop]
doc14_last = tokenizer.tokenize((' '.join(doc14_new)).lower())
doc14 = doc14_last
doc15_new = [j for j in doc15 if j not in stop]
doc15_last = tokenizer.tokenize((' '.join(doc15_new)).lower())
doc15 = doc15_last

docs = [doc2, doc3, doc4, doc6, doc8, doc9, doc10, doc11, doc12, doc14];
def var_em(K,M,N,doc_list):
	for d in range(M):
		doc_change = doc_list[d]
		doc_list[d] = doc_change[:N]

	voc = {}
	count = 0
	for d in doc_list:
		for n in d:
			if (n in voc)==False:
				voc[n] = count
				count = count + 1
	vocabulary = voc.keys()
	
	#initializations
	alfa = np.array([50.0/K]*K)

	#gamma = np.array([alfa[0]+1.0*N/K]*K)
	gamma = np.array([])
	for i in range(K):
		gamma = np.append(gamma,[alfa[0]+random.uniform(0.9,1.0)*N/K])
	Gamma = np.array([gamma])
	for d in range(M-1):
		gamma = np.array([])
		for i in range(K):
			gamma = np.append(gamma,[alfa[0]+random.uniform(0.9,1.0)*N/K])
		Gamma = np.append(Gamma,[gamma],axis=0)
	print Gamma

	#beta = np.array([1.0/len(vocabulary)]*len(vocabulary))
	beta = np.array([])
	for i in range(len(vocabulary)):
		beta = np.append(beta,[random.random()])
	beta = beta/sum(beta)
	Beta = np.array([beta])
	for i in range(K-1):
		beta = np.array([])
		for i in range(len(vocabulary)):
			beta = np.append(beta,[random.random()])
		beta = beta/sum(beta)
		Beta = np.append(Beta,[beta],axis=0)

	#faydoc = np.array([1.0/K]*K)
	faydoc = np.array([])
	for i in range(K):
		faydoc = np.append(faydoc,[random.random()])
	faydoc = faydoc/sum(faydoc)
	fay_doc = np.array([faydoc])
	for n in range(N-1):
		faydoc = np.array([])
		for i in range(K):
			faydoc = np.append(faydoc,[random.random()])
		faydoc = faydoc/sum(faydoc)
		fay_doc = np.append(fay_doc,[faydoc],axis=0)
	Fay = np.array([fay_doc])
	for d in range(M-1):
		faydoc = np.array([])
		for i in range(K):
			faydoc = np.append(faydoc,[random.random()])
		faydoc = faydoc/sum(faydoc)
		fay_doc = np.array([faydoc])
		for n in range(N-1):
			faydoc = np.array([])
			for i in range(K):
				faydoc = np.append(faydoc,[random.random()])
			faydoc = faydoc/sum(faydoc)
			fay_doc = np.append(fay_doc,[faydoc],axis=0)
		Fay = np.append(Fay,[fay_doc],axis=0)
	
	count = 0
	while count<7:
		#E-step
		for d in range(M):
			say = 0
			while say<15:
				for n in range(N):
					for i in range(K):
						#voc_place = vocabulary.index(docs[d][n])
						Fay[d][n][i] = Beta[i][voc[docs[d][n]]]*math.exp(sc.special.digamma(Gamma[d][i]))
					Fay[d][n] = Fay[d][n]/sum(Fay[d][n])
				Gamma[d] = alfa + np.sum(Fay[d],axis=0)
				say = say+1
		print Fay
		print Gamma
		#M-step
		for i in range(K):
			for j in range(len(vocabulary)):
				tot = 0
				for d in range(M):
					for n in range(N):
						if voc[docs[d][n]] == j:
							tot = tot + Fay[d][n][i]
				Beta[i][j] = tot
			Beta[i] = Beta[i]/sum(Beta[i])
		print Beta
		loc_gam = 0
		for d in range(M):
			loc_gam = loc_gam + sc.special.digamma(Gamma[d][0]) - sc.special.digamma(sum(Gamma[d]))
		dLda = M*(K*sc.special.polygamma(1, K*alfa[0])-K*sc.special.polygamma(1, alfa[0])) + loc_gam
		d2Lda2 = M*(K*K*sc.special.polygamma(2, K*alfa[0]) - K*sc.special.polygamma(2, alfa[0]))
		alfa[0] = math.exp(math.log(alfa[0])-dLda/(d2Lda2*alfa[0]+dLda))
		alfa[1] = alfa[0]
		print alfa
		count = count + 1

	beta0 = Beta[0]
	ind0 = sorted(range(len(beta0)), key=lambda k: -beta0[k])
	for i in range(40):
		print voc.keys()[voc.values().index(ind0[i])]
	print 'TRABZONSPOR 50.YILDA SAMPIYON'
	beta1 = Beta[1]
	ind1 = sorted(range(len(beta1)), key=lambda k: -beta1[k])
	for i in range(40):
		print voc.keys()[voc.values().index(ind1[i])]

	

var_em(2,10,1000,docs)

