"""Metrics to compute the model performance."""
from DataParser import DataParser as DataParser
from model7 import Model7 as Model
import numpy as np
import os, sys
import math
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer

def ComputePrecisionK(modelfile,testfile,outputfile):
    maxParagraphLength = int(sys.argv[1])
    maxParagraphs = int(sys.argv[2] )
    filterSizes = [int(i) for i in sys.argv[3].split("-")]
    num_filters = int(sys.argv[4])
    wordEmbeddingDimension = int(sys.argv[5])
    lrate = float(sys.argv[10])
    poolLength = int(sys.argv[11])

    labels = 30938
    vocabularySize = 101939

    model = Model(maxParagraphs,maxParagraphLength,labels,vocabularySize,filterSizes,num_filters,wordEmbeddingDimension,lrate, poolLength)

    testing = DataParser(maxParagraphs,maxParagraphLength,labels,vocabularySize)
    testing.getDataFromfile(testfile)

    model.load(modelfile)

    print("loading done")
    print("no of test examples: " + str(testing.totalPages))

    print("Computing Prec@k")
    
    #check if batchsize needs to be taken by parameter

    batchSize = 1
    testing.restore()
    truePre=[]
    pred=[]
    for itr in range(testing.totalPages):
        data=testing.nextBatch(1)
        truePre.append(data[0])
        pre=model.predict(data)
        pred.append(pre[0])
    
    k = 5
    for i in [1,3,5] :
        val = ndcg_score( truePre, pred , i)
        print( val )

    def dcg_score(y_true, y_score, k=5):
        """Discounted cumulative gain (DCG) at rank K.

        Parameters
        ----------
        y_true : array, shape = [n_samples]
            Ground truth (true relevance labels).
        y_score : array, shape = [n_samples, n_classes]
            Predicted scores.
        k : int
            Rank.

        Returns
        -------
        score : float
        """
        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order[:k])

        gain = 2 ** y_true - 1

        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(gain / discounts)


    def ndcg_score(ground_truth, predictions, k=5):
        """Normalized discounted cumulative gain (NDCG) at rank K.

        Normalized Discounted Cumulative Gain (NDCG) measures the performance of a
        recommendation system based on the graded relevance of the recommended
        entities. It varies from 0.0 to 1.0, with 1.0 representing the ideal
        ranking of the entities.

        Parameters
        ----------
        ground_truth : array, shape = [n_samples]
            Ground truth (true labels represended as integers).
        predictions : array, shape = [n_samples, n_classes]
            Predicted probabilities.
        k : int
            Rank.

        Returns
        -------
        score : float

        Example
        -------
        >>> ground_truth = [1, 0, 2]
        >>> predictions = [[0.15, 0.55, 0.2], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
        >>> score = ndcg_score(ground_truth, predictions, k=2)
        1.0
        >>> predictions = [[0.9, 0.5, 0.8], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
        >>> score = ndcg_score(ground_truth, predictions, k=2)
        0.6666666666
        """
        lb = LabelBinarizer()
        lb.fit(range(len(predictions) + 1))
        T = lb.transform(ground_truth)

        scores = []

        # Iterate over each y_true and compute the DCG score
        for y_true, y_score in zip(T, predictions):
            actual = dcg_score(y_true, y_score, k)
            best = dcg_score(y_true, y_true, k)
            score = float(actual) / float(best)
            scores.append(score)

        return np.mean(scores)


