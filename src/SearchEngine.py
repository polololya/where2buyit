import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as sl
import time
import matplotlib.pyplot as plt
import cv2
import math
# --------главная функция--------
def SimilarItem(target_embeding, dataset_embedings,K):
    value_similarity = pd.DataFrame(
        np.squeeze(sl.cosine_similarity(target_embeding.values.reshape(1, -1), dataset_embedings)))
    # dataset_embedings.insert(0,'similar',value_similarity) - не делает копию объекта и удалить этот столбец тяжело
    dataset_embedings = dataset_embedings.assign(similar=value_similarity)  # создает дубликат
    dataset_embedings = pd.DataFrame(dataset_embedings).sort_values(by='similar', ascending=False)
    similarityK = dataset_embedings.iloc[:K]
    return similarityK

# --------метрики--------
PrecisionK=lambda similarityK_index,positive_index:len(set(similarityK_index)&set(positive_index))/len(similarityK_index)
def AvgPrecisionK(target_embedings,dataset_embedings,positive_indices,K):
    value_similarity=pd.DataFrame(np.squeeze(sl.cosine_similarity(target_embedings, dataset_embedings)))
    similarityK_indices=pd.DataFrame(np.argsort(value_similarity)).iloc[:,-K:]#получаем индексы
    n = len(target_embedings)
    k = np.zeros((n))
    for i in range(n):
        k[i]=PrecisionK(similarityK_indices.iloc[i], positive_indices.iloc[i])
    k=sum(k)/n
    #по идее 24-27 быстрее, если target'ов много
    #k=np.array(list(map(lambda i: PrecisionK1(similarity5.iloc[i],positive_indices[i]),similarity5.index)))
    return k
def similar_visual(similarityK,column,DATASET_DIR=''):#принимает эмбединги похожих и столбец в котором хранится путь к изображениям
    pic_box = plt.figure()
    i=1
    n=len(similarityK)
    col = int(math.sqrt(n))
    row = math.ceil(n / col)
    for value in similarityK.loc[:, column]:
        #imread не читает путь с русскими буквами
        picture = cv2.imread(value)
        # конвертируем BGR изображение в RGB
        picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
        # добавляем ячейку в pix_box для вывода текущего изображения
        pic_box.add_subplot(row, col, i)
        plt.imshow(picture)
        # отключаем отображение осей
        plt.axis('off')
        i+=1
    pic_box.tight_layout(pad=0, w_pad=0, h_pad=0)
    # выводим все созданные фигуры на экран
    plt.show()
