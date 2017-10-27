# -*- coding: utf-8 -*-
import pandas as pd
import random as rd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

rd.seed(a=None)

def readDataFile(filePath):
    # Carregando dados
    dataSet = pd.read_table(filePath,
                            sep=",")
    
    print("Total de dados antes do tratamento:" + str(len(dataSet)))
    
    # Fazendo tratamento dos dados
    dataSet.columns = ['b', 'g', 'r', 'class']
    dataSet = dataSet.replace({ '?':'-1' })
    dataSet = dataSet.astype(float)
    dataSet = dataSet.replace({ -1:np.NaN })
    dataSet = dataSet.dropna()
    
    dataSet = dataSet.loc[((dataSet['b'] <= 255.0) & (dataSet['g'] <= 255.0) & (dataSet['r'] <= 255.0))]
    
    print("Total de dados após o tratamento:" + str(len(dataSet)))
    
    return dataSet

def getDataSets(data, trainFactor=0.9):
    # Tamanho do conjunto de treinamento
    trainSetCount = int(round(len(data) * trainFactor))
    # Seleciona aleatoriamente o conjunto de treinamento
    trainSetIndexList = rd.sample(range(0, len(data) - 1), trainSetCount)
    
    # Cria o conjunto de treinamento a partir do sorteio aleatório
    trainSet = data.loc[data.index[trainSetIndexList]]
    trainSet = trainSet.reset_index()
    # Cria o conjunto de teste com o restante dos dados
    testSet = data.drop(data.index[trainSetIndexList])
    testSet = testSet.reset_index()
    
    return trainSet, testSet

def convertToYCrCb(dataSet):
    yCrCbDataset = pd.DataFrame(columns=['y', 'cr', 'cb', 'class'])

    for i in range(0, len(dataSet)):
        y = 0.299 * dataSet['r'].values[i] + 0.587 * dataSet['g'].values[i] + 0.114 * dataSet['b'].values[i]
        cr = ((dataSet['r'].values[i] - y) / 1.6) + 0.5
        cb = ((dataSet['b'].values[i] - y) / 2.0) + 0.5
        
        yCrCbDataset = yCrCbDataset.append([{'y':float(y), 'cr':float(cr), 'cb':float(cb), 'class':float(dataSet['class'].values[i])}], ignore_index=True)
        
    return yCrCbDataset

data_rgb = readDataFile("database-pele-07.dat")

model_test_rgb = pd.DataFrame(columns=['L', 'RL', 'NB', 'NN'])
model_test_ycrcb = pd.DataFrame(columns=['L', 'RL', 'NB', 'NN'])

for i in range(0, 10):
    # ========== RGB ========== #
    trainSet_rgb, testSet_rgb = getDataSets(data_rgb)
    
    trainSet_x_rgb = trainSet_rgb[['b', 'g', 'r']]
    trainSet_y_rgb = np.ravel(trainSet_rgb[['class']])
    
    testSet_x_rgb = testSet_rgb[['b', 'g', 'r']]
    testSet_y_rgb = np.ravel(testSet_rgb[['class']])
    
    l_model_rgb = LinearRegression()
    l_model_rgb.fit(trainSet_x_rgb, trainSet_y_rgb)
    
    rl_model_rgb = LogisticRegression()
    rl_model_rgb.fit(trainSet_x_rgb, trainSet_y_rgb)
    
    nb_model_rgb = GaussianNB()
    nb_model_rgb.fit(trainSet_x_rgb, trainSet_y_rgb)
    
    nn_model_rgb = KNeighborsClassifier()
    nn_model_rgb.fit(trainSet_x_rgb, trainSet_y_rgb)
    
    # ========== END RGB ========== #
    
    # ========== YCrCb ========== #
    
    trainSet_ycrcb = convertToYCrCb(trainSet_rgb)
    testSet_ycrcb = convertToYCrCb(testSet_rgb)
    
    trainSet_x_ycrcb = trainSet_ycrcb[['cr', 'cb']]
    trainSet_y_ycrcb = np.ravel(trainSet_ycrcb[['class']])
    
    testSet_x_ycrcb = testSet_ycrcb[['cr', 'cb']]
    testSet_y_ycrcb = np.ravel(testSet_ycrcb[['class']])
    
    l_model_ycrcb = LinearRegression()
    l_model_ycrcb.fit(trainSet_x_ycrcb, trainSet_y_ycrcb)
    
    rl_model_ycrcb = LogisticRegression()
    rl_model_ycrcb.fit(trainSet_x_ycrcb, trainSet_y_ycrcb)
    
    nb_model_ycrcb = GaussianNB()
    nb_model_ycrcb.fit(trainSet_x_ycrcb, trainSet_y_ycrcb)
    
    nn_model_ycrcb = KNeighborsClassifier()
    nn_model_ycrcb.fit(trainSet_x_ycrcb, trainSet_y_ycrcb)
    
    # ========== END YCrCb ========== #
    
    l_rgb_score = l_model_rgb.score(testSet_x_rgb, testSet_y_rgb)
    rl_rgb_score = rl_model_rgb.score(testSet_x_rgb, testSet_y_rgb)
    nb_rgb_score = nb_model_rgb.score(testSet_x_rgb, testSet_y_rgb)
    nn_rgb_score = nn_model_rgb.score(testSet_x_rgb, testSet_y_rgb)
    
    model_test_rgb = model_test_rgb.append([{
            'L':float(l_rgb_score), 
            'RL':float(rl_rgb_score), 
            'NB':float(nb_rgb_score), 
            'NN':float(nn_rgb_score)}], 
        ignore_index=True)
    
    l_ycrcb_score = l_model_ycrcb.score(testSet_x_ycrcb, testSet_y_ycrcb)
    rl_ycrcb_score = rl_model_ycrcb.score(testSet_x_ycrcb, testSet_y_ycrcb)
    nb_ycrcb_score = nb_model_ycrcb.score(testSet_x_ycrcb, testSet_y_ycrcb)
    nn_ycrcb_score = nn_model_ycrcb.score(testSet_x_ycrcb, testSet_y_ycrcb)
    
    model_test_ycrcb = model_test_ycrcb.append([{
            'L':float(l_ycrcb_score), 
            'RL':float(rl_ycrcb_score), 
            'NB':float(nb_ycrcb_score), 
            'NN':float(nn_ycrcb_score)}], 
        ignore_index=True)
    
    print()
    
    print("Test:" + str(i))
    
    print({"L_Score_rgb":l_rgb_score})
    #print({"RL_Coef._rgb":rl_model_rgb.coef_})
    print({"RL_Score_rgb":rl_rgb_score})
    print({"NB_Score_rgb":nb_rgb_score})
    print({"NN_Score_rgb":nn_rgb_score})
    
    print()
    
    print({"L_Score_ycrcb":l_ycrcb_score})
    #print({"RL_Coef._ycrcb":rl_model_ycrcb.coef_})
    print({"RL_Score_ycrcb":rl_ycrcb_score})
    print({"NB_Score_ycrcb":nb_ycrcb_score})
    print({"NN_Score_ycrcb":nn_ycrcb_score})
    
print(model_test_rgb.describe())
print(model_test_ycrcb.describe())

del data_rgb, trainSet_rgb, testSet_rgb, trainSet_x_rgb, trainSet_y_rgb, testSet_x_rgb, testSet_y_rgb, rl_model_rgb
del trainSet_ycrcb, testSet_ycrcb, trainSet_x_ycrcb, trainSet_y_ycrcb, testSet_x_ycrcb, testSet_y_ycrcb, rl_model_ycrcb