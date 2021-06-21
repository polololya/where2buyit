import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as sl
import time
# --------главная функция--------
def SimilarItem(target_embeding, dataset_embedings):
    value_similarity = pd.DataFrame(
        np.squeeze(sl.cosine_similarity(target_embeding.values.reshape(1, -1), dataset_embedings)))
    # dataset_embedings.insert(0,'similar',value_similarity) - не делает копию объекта и удалить этот столбец тяжело
    dataset_embedings = dataset_embedings.assign(similar=value_similarity)  # создает дубликат
    dataset_embedings = pd.DataFrame(dataset_embedings).sort_values(by='similar', ascending=False)
    similarity5 = dataset_embedings.iloc[:5]
    return similarity5


# --------метрики--------
PrecisionK=lambda similarity5_index,positive_index: len(set(similarity5_index)&set(positive_index))/len(similarity5_index)
#PrecisionK работает только для одного целевого эмбеддинга, принимает вторым значением индексы реальных похожих предметов
def AvgPrecisionK(target_embedings,dataset_embedings,positive_indices):
    value_similarity=pd.DataFrame(np.squeeze(sl.cosine_similarity(target_embedings, dataset_embedings)))
    similarity5=pd.DataFrame(np.argsort(value_similarity)).iloc[:,-5:]
    n = len(target_embedings)
    k = np.zeros((n))
    for i in range(n):
        k[i]=PrecisionK(similarity5.iloc[i], positive_indices[i])
    k=sum(k)/n
    #по идее 24-27 быстрее, если target'ов много
    #k=np.array(list(map(lambda i: PrecisionK1(similarity5.iloc[i],positive_indices[i]),similarity5.index)))
    return k
emb_size = 256
dataset_size = 100

df = pd.DataFrame(np.random.rand(dataset_size, emb_size))

classes = ['cat','dog','horse','pig','hen']
df['label'] = np.random.choice(classes, size=dataset_size)

df['img_path'] = np.random.randint(low=100, high=1000, size=dataset_size).astype(str)
df['img_path'] = 'C://dummy_folder/' + df['img_path'] + '.png'

target=df.iloc[np.random.randint(0,len(df)),:emb_size]
start_time = time.time()
dataset=df.iloc[:,:emb_size]
similarity5=SimilarItem(target, dataset)
similarity5=similarity5.join(df.iloc[similarity5.index,emb_size:])
print("--- %s seconds ---" % (time.time() - start_time))
print(df.iloc[similarity5.index,:])
print(similarity5)
print("Столбцы не потерялись, SimilarItem работает, если подавать только значения без текста ")
positive_index=np.random.choice(dataset.index,np.random.randint(1,len(dataset)),replace = False)
#print(positive_index)
start_time = time.time()
K=PrecisionK(similarity5.index,positive_index)
print("--- %s seconds ---" % (time.time() - start_time))
print(K, ' - значение метрики для одного целевого изображения')
targets=df.iloc[np.random.randint(len(df),size=np.random.randint(len(df))),:emb_size]#рандомные целевые векторы
positive_indices=[]# индексы похожих изображений из базы для каждого целевого изображения в разных количествах
for i in range (len(targets)):
    positive_indices+=[np.random.choice(dataset.index,np.random.randint(1,len(dataset)),replace = False)]
positive_indices=pd.DataFrame(positive_indices)
start_time = time.time()
k=AvgPrecisionK(targets,dataset,positive_indices)
print("--- %s seconds ---" % (time.time() - start_time))
print(k,' - усредненное значение метрики')
