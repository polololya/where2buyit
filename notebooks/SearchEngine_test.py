from SearchEngine import *
emb_size = 4
dataset_size = 90

df = pd.DataFrame(np.random.rand(dataset_size, emb_size))

classes = ['cat','dog','horse','pig','hen']
df['label'] = np.random.choice(classes, size=dataset_size)

df['img_path'] = np.random.randint(low=3, high=90, size=dataset_size).astype(str)
df['img_path'] = r'C:\Users\user\Desktop\imp\stag2\pici\Screenshot_' + df['img_path'] + '.png'
K=5
print(df)
target=df.iloc[np.random.randint(0,len(df)),:emb_size]
start_time = time.time()
dataset=df.iloc[:,:emb_size]
similarity5=SimilarItem(target, dataset,K)
similarity5=similarity5.join(df.iloc[similarity5.index,emb_size:])
print("--- %s seconds ---" % (time.time() - start_time))
print(df.iloc[similarity5.index,:])
print(similarity5)
print("Столбцы не потерялись, SimilarItem работает, если подавать только значения без текста ")
positive_index=np.random.choice(dataset.index,np.random.randint(1,len(dataset)),replace = False)
print(positive_index)
#print(positive_index)
start_time = time.time()
k=PrecisionK(similarity5.index,positive_index)
print("--- %s seconds ---" % (time.time() - start_time))
print(k, ' - значение метрики для одного целевого изображения')
targets=df.iloc[np.random.randint(len(df),size=np.random.randint(len(df))),:emb_size]#рандомные целевые векторы
positive_indices=[]# индексы похожих изображений из базы для каждого целевого изображения в разных количествах
for i in range (len(targets)):
    positive_indices+=[np.random.choice(dataset.index,np.random.randint(1,len(dataset)),replace = False)]
    print(SimilarItem(targets.iloc[i], dataset,K).index)
positive_indices=pd.DataFrame(positive_indices)
print(positive_indices)
start_time = time.time()
k=AvgPrecisionK(targets,dataset,positive_indices,K)
print("--- %s seconds ---" % (time.time() - start_time))
print(k,' - усредненное значение метрики')
similar_visual(similarity5,'img_path')