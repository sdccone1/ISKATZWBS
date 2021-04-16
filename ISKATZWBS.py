import numpy as np
import math
import time
import csv

# 高斯普核相似度
def GaussianInteractionProfileKernelSimilarity(biAdjMatrix):
    # 求权重
    widthSum = 0
    for i in range(biAdjMatrix.shape[0]):
        width = np.linalg.norm(biAdjMatrix[i])  # 求行向量的二范数,这里注意如果手动实现的话Python中的a[i][j] 要用 a[i, j]来表示
        widthSum += width ** 2
    r = biAdjMatrix.shape[0] / widthSum
    similarityMatrix = np.mat(np.zeros((biAdjMatrix.shape[0], biAdjMatrix.shape[0])))
    for i in range(biAdjMatrix.shape[0]):
        for j in range(i, biAdjMatrix.shape[0]):
            ip = 0
            ip += (np.linalg.norm(biAdjMatrix[i, :] - biAdjMatrix[j, :])) ** 2
            similarityMatrix[i, j] = math.exp(-r * ip)
            similarityMatrix[j, i] = similarityMatrix[i, j]
    return similarityMatrix

#cosine similarity
def CosineSimilarity(biAdjMatrix):
    similarityMatrix = np.mat(np.zeros((biAdjMatrix.shape[0], biAdjMatrix.shape[0])))
    for i in range(biAdjMatrix.shape[0]):
        for j in range(i, biAdjMatrix.shape[0]):
            if (np.linalg.norm(biAdjMatrix[i, :]) * np.linalg.norm(biAdjMatrix[j, :])) == 0:
                similarityMatrix[i, j] = 0
            else:
                similarityMatrix[i, j] = np.dot(biAdjMatrix[i, :], biAdjMatrix[j, :].T) / (np.linalg.norm(biAdjMatrix[i, :]) * np.linalg.norm(biAdjMatrix[j, :]))
            similarityMatrix[j, i] = similarityMatrix[i, j]
    return similarityMatrix

#整合新的相似度矩阵
def IntegerateSimilarityMatrix(kS, CS, w):
    similarityMatrix = w * kS + (1-w) * CS
    return similarityMatrix

# WBSMDA
def WBSMDA(biAdjMatrix, similarityPathogen, similarityHost):
    resultMatrix = np.full(biAdjMatrix.shape, np.nan)
    for i in range(biAdjMatrix.shape[0]):
        for j in range(biAdjMatrix.shape[1]):
            cmw = np.nan
            cdw = np.nan
            cmb = np.nan
            cdb = np.nan
            for k in range(similarityPathogen.shape[0]):
                if k == i:
                    continue
                if int(biAdjMatrix[k, j]) == 1 and (cmw is np.nan or cmw < similarityPathogen[i, k]):
                    cmw = similarityPathogen[i, k]
                elif int(biAdjMatrix[k, j]) == 0 and (cmb is np.nan or cmb < similarityPathogen[i, k]):
                    cmb = similarityPathogen[i, k]
            for k in range(similarityHost.shape[0]):
                if k == j:
                    continue
                if int(biAdjMatrix[i, k]) == 1 and (cdw is np.nan or cdw < similarityHost[j, k]):
                    cdw = similarityHost[j, k]
                elif int(biAdjMatrix[i, k]) == 0 and (cdb is np.nan or cdb < similarityHost[j, k]):
                    cdb = similarityHost[j, k]
            if cdw is np.nan:
                resultMatrix[i, j] = cmw / cmb
            elif cmw is np.nan:
                resultMatrix[i, j] = cdw / cdb
            else:
                resultMatrix[i, j] = (cmw * cdw) / (cmb * cdb)
    return resultMatrix

def IMKATZWBSMDA(biAdjMatrix, row=None, col=None):
    kp = GaussianInteractionProfileKernelSimilarity(biAdjMatrix)
    kh = GaussianInteractionProfileKernelSimilarity(biAdjMatrix.T)
    cp = CosineSimilarity(biAdjMatrix)
    ch = CosineSimilarity(biAdjMatrix.T)
    pathogenSimilarity = IntegerateSimilarityMatrix(kp, cp, 0.5)
    hostSimilarity = IntegerateSimilarityMatrix(kh, ch, 0.5)

    WBSMDAMatrix = WBSMDA(biAdjMatrix, pathogenSimilarity, hostSimilarity)

    resultRow = []
    for i in range(11):
        for j in range(11):
            a = float(i / 10)
            b = float(j / 10)

            S = 0.01 * WBSMDAMatrix + 0.0001 * (a * WBSMDAMatrix * hostSimilarity + b * pathogenSimilarity * WBSMDAMatrix)
            if row is not None and col is not None:
                resultRow.append(S[row, col])
            else:
                np.save('./a=' + str(a) + ' b=' + str(b) + '.npy', S)

    if row is not None and col is not None:
        return resultRow


biAdjMatrix = np.mat(np.load('../data/biAdjMatrix.npy'))
IMKATZWBSMDA(biAdjMatrix)

sumCnt = 0
for i in range(biAdjMatrix.shape[0]):
    for j in range(biAdjMatrix.shape[1]):
        if int(biAdjMatrix[i, j]) == 1:
            sumCnt += 1

BKATZWBSMDAResultTable = []
cnt = 1
for i in range(biAdjMatrix.shape[0]):
    for j in range(biAdjMatrix.shape[1]):
        if int(biAdjMatrix[i, j]) == 0:
            continue
        biAdjMatrix[i, j] = 0
        BKATZWBSMDAResultRow = IMKATZWBSMDA(biAdjMatrix, i, j)
        BKATZWBSMDAResultTable.append(BKATZWBSMDAResultRow)
        biAdjMatrix[i, j] = 1
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        print('已完成' + str(float(cnt / sumCnt) * 100) + '%')
        cnt += 1

with open('./LOOCV.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for row in BKATZWBSMDAResultTable:
        writer.writerow(row)

print('完成！')
