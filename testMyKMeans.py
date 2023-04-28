# 谨慎运行 会批量处理data/gt下图片 保存至data/myKMeans

from myKMeans import myKMeans
import numpy as np
import os
import glob
from PIL import Image


# 测试 myKMeans 类型
# 输入图像路径
# 结果图片保存至saveFolder
def testMyKMeans(path='', k=4, saveFolder=''):
    fileList = glob.glob(os.path.join(path, "*"))
    labelList = []
    for filePath in fileList:
        img = Image.open(filePath)
        img = np.asarray(img)
        pixels = img.reshape((-1, 3))
        mykmeans = myKMeans(k,distanceType='manhattan')
        mykmeans.fit(pixels)
        labels = mykmeans.predict(pixels)
        centers = mykmeans.centers
        labels = labels.reshape(img.shape[:2])
        labelList.append(labels)
        segmentedImg = Image.fromarray(np.uint8(centers[labels]))
        fileName = os.path.basename(filePath)
        savePath = os.path.join(saveFolder, fileName)
        segmentedImg.save(savePath)
    return labelList


labelList = testMyKMeans('data/imgs', k=2, saveFolder='data/myKMeans')
