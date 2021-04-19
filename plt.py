import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import csv

def getAUC(matrix, array, biAdjMatrix):
    negScoreList = []
    for m in range(biAdjMatrix.shape[0]):
        for n in range(biAdjMatrix.shape[1]):
            if int(biAdjMatrix[m, n]) == 0:
                negScoreList.append(matrix[m, n])
    y_score = array[~np.isnan(array)]
    y_neg = np.array(negScoreList)
    y_neg = y_neg[~np.isnan(y_neg)]
    y_true = np.concatenate([np.ones(y_score.shape[0]), np.zeros(y_neg.shape[0])])
    y_score = np.concatenate([y_score, y_neg])
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return fpr, tpr, auc

def getPR(matrix, array, biAdjMatrix):
    negScoreList = []
    for m in range(biAdjMatrix.shape[0]):
        for n in range(biAdjMatrix.shape[1]):
            if int(biAdjMatrix[m, n]) == 0:
                negScoreList.append(matrix[m, n])
    y_score = array[~np.isnan(array)]
    y_neg = np.array(negScoreList)
    y_neg = y_neg[~np.isnan(y_neg)]
    y_true = np.concatenate([np.ones(y_score.shape[0]), np.zeros(y_neg.shape[0])])
    y_score = np.concatenate([y_score, y_neg])
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score)
    aupr = metrics.auc(recall, precision)
    return precision, recall, aupr


biAdjMatrix = np.mat(np.load('./data/biAdjMatrix.npy'))

fprList = []
tprList = []
aucList = []
precisionList = []
recallList = []
auprList = []

# IMKATZWBS

arrayList = []
predictResult = []
for i in range(121):
    predictResult.append([])
with open('./IMKATZWBS/LOOCV.csv') as f:
    csvReader = csv.reader(f)
    for row in csvReader:
        for i in range(121):
            predictResult[i].append(float(row[i]))
arrayList = predictResult[22]
f, t, a = getAUC(np.load('./IMKATZWBS/a=0.2 b=0.0.npy'), np.array(arrayList), biAdjMatrix)
fprList.append(f)
tprList.append(t)
aucList.append(a)
p, r, a = getPR(np.load('./IMKATZWBS/a=0.2 b=0.0.npy'), np.array(arrayList), biAdjMatrix)
precisionList.append(p)
recallList.append(r)
auprList.append(a)

# KATZHMDA
f, t, a = getAUC(np.mat(np.load('./KATZHMDA/result.npy')), np.array(np.load('./KATZHMDA/LOOCV.npy')), biAdjMatrix)
fprList.append(f)
tprList.append(t)
aucList.append(a)
p, r, a = getPR(np.mat(np.load('./KATZHMDA/result.npy')), np.array(np.load('./KATZHMDA/LOOCV.npy')), biAdjMatrix)
precisionList.append(p)
recallList.append(r)
auprList.append(a)

# WBSMDA
f, t, a = getAUC(np.mat(np.load('./WBSMDA/result.npy')), np.array(np.load('./WBSMDA/LOOCV.npy')), biAdjMatrix)
fprList.append(f)
tprList.append(t)
aucList.append(a)
p, r, a = getPR(np.mat(np.load('./WBSMDA/result.npy')), np.array(np.load('./WBSMDA/LOOCV.npy')), biAdjMatrix)
precisionList.append(p)
recallList.append(r)
auprList.append(a)

# NGRHMDA
f, t, a = getAUC(np.mat(np.load('./NGRHMDA/result.npy')), np.array(np.load('./NGRHMDA/LOOCV.npy')), biAdjMatrix)
fprList.append(f)
tprList.append(t)
aucList.append(a)
p, r, a = getPR(np.mat(np.load('./NGRHMDA/result.npy')), np.array(np.load('./NGRHMDA/LOOCV.npy')), biAdjMatrix)
precisionList.append(p)
recallList.append(r)
auprList.append(a)

# NCPHMDA
f, t, a = getAUC(np.mat(np.load('./NCPHMDA/result.npy')), np.array(np.load('./NCPHMDA/LOOCV.npy')), biAdjMatrix)
fprList.append(f)
tprList.append(t)
aucList.append(a)
p, r, a = getPR(np.mat(np.load('./NCPHMDA/result.npy')), np.array(np.load('./NCPHMDA/LOOCV.npy')), biAdjMatrix)
precisionList.append(p)
recallList.append(r)
auprList.append(a)


# 输出ROC曲线图像
with plt.style.context(['science', 'ieee', 'no-latex']):
    plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    nameList = ['ISKATZWBS AUC=%.3f' % aucList[0], 'KATZHMDA AUC=%.3f' % aucList[1],
            'WBSMDA AUC=%.3f' % aucList[2], 'NGRHMDA AUC=%.3f' % aucList[3]]
    colorList = ['navy', 'aqua', 'darkorange', 'cornflowerblue', 'red']
    # plt.plot(fprList[0], tprList[0], color='navy', linewidth=2, label='BKATZWBS AUC=%.3f' % aucList[0])
    for i, name, color in zip(range(len(fprList)), nameList, colorList):
        plt.plot(fprList[i], tprList[i], color=color, label=name)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('HMDAD数据集预测ROC曲线和对应AUC值')
    plt.legend(loc="lower right")
    plt.show()

# 输出PR曲线图像
with plt.style.context(['science', 'ieee', 'no-latex']):
    plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    nameList = ['ISKATZWBS AUPR=%.3f' % auprList[0], 'KATZHMDA AUPR=%.3f' % auprList[1],
            'WBSMDA AUPR=%.3f' % auprList[2], 'NGRHMDA AUPR=%.3f' % auprList[3]]
    colorList = ['navy', 'aqua', 'darkorange', 'cornflowerblue', 'red']
#plt.plot(recallList[0], precisionList[0], color='navy', linewidth=2, label='BKATZWBS AUPR=%.3f' % auprList[0])
    for i, name, color in zip(range(len(fprList)), nameList, colorList):
        plt.plot(recallList[i], precisionList[i], color=color, label=name)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower right")
    plt.show()
