# 比照ground truth与myKMeans分割的图片
# 以 精确度、 召回度 、 F1值 为标准
# 评估结果可视化

import numpy as np
import cv2
import glob
import os
from sklearn.metrics.cluster import contingency_matrix
import matplotlib.pyplot as plt


# 输入混淆矩阵计算精确度、 召回值、 F1值
def Measure(ConfusionMatrix):
    totalF1Score = 0
    totalPrecision = 0
    totalRecall = 0
    num = len(ConfusionMatrix[0])
    # Evaluate each label in Confusion Matrix
    for i in range(num):
        maxSample = np.argmax(ConfusionMatrix[:, i])
        sampleSum = np.sum(ConfusionMatrix[:, i])
        precision = ConfusionMatrix[maxSample][i] / sampleSum
        recall = ConfusionMatrix[maxSample][i] / np.sum(ConfusionMatrix[maxSample, :])
        F1Score = (2 * precision * recall) / (precision + recall)
        totalPrecision += precision
        totalRecall += recall
        totalF1Score += F1Score
    return totalPrecision / num, totalRecall / num, totalF1Score / num


# 输入： 背景路径、 待评测图片路径
# 输出： 精确度、 召回值、 F1值
def evaluatePair(gtPath, resPath):
    gt = cv2.imread(gtPath)
    res = cv2.imread(resPath)
    gt = cv2.threshold(gt, 100, 255, cv2.THRESH_BINARY)[1]
    res = cv2.threshold(res, res.mean(), 255, cv2.THRESH_BINARY)[1]
    gt = gt.reshape(-1)
    res = res.reshape(-1)
    contigencyMatrix = contingency_matrix(gt, res)
    (p, r, f) = Measure(contigencyMatrix)
    return p, r, f


# 输入： 背景图片所在文件夹、 待评测图片所在文件夹、背景图片的文件类型后缀
# 待评测对的文件名相同，后缀可以不同
# 输出： List of 精确度、 召回值、 F1值
def evaluateFolder(gtFolder, resFolder, gtSuffix='.png'):
    precisionList = []
    recallList = []
    F1List = []

    resList = glob.glob(os.path.join(resFolder, "*"))
    for resPath in resList:
        resName = os.path.basename(resPath)
        resBase, resSuffix = os.path.splitext(resName)
        gtName = resBase + gtSuffix
        gtPath = os.path.join(gtFolder, gtName)
        (p, r, f) = evaluatePair(gtPath, resPath)
        precisionList.append(p)
        recallList.append(r)
        F1List.append(f)
    return precisionList, recallList, F1List


gtFolder = 'data/gt'
resFolder = 'data/myKMeansEuclidean'

precisionList, recallList, F1List = evaluateFolder(gtFolder, resFolder)

# 各图像 测试结果
numImages = len(precisionList)
indices = np.arange(numImages)
width = 0.2
fig, ax = plt.subplots()
rects1 = ax.bar(indices - width, precisionList, width, label='Precision')
rects2 = ax.bar(indices, recallList, width, label='Recall')
rects3 = ax.bar(indices + width, F1List, width, label='F1 Score')
ax.set_xlabel('Image Number')
ax.set_ylabel('Evaluation Metric')
ax.set_title('MyKMeans Segmentation Performance')
ax.set_xticks(indices)
ax.legend()

# 平均数据
avgPrecision = np.mean(precisionList)
avgRecall = np.mean(recallList)
avgF1 = np.mean(F1List)
fig, ax = plt.subplots()
metrics = ('Precision', 'Recall', 'F1 Score')
values = (avgPrecision, avgRecall, avgF1)

ax.bar(metrics, values)
ax.set_xlabel('Evaluation Metric')
ax.set_ylabel('Average Value')
ax.set_title('MyKMeans Segmentation Performance (Average)')

plt.show()
