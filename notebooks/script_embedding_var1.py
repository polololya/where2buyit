from HardTripletLoss import EmbeddingModel
import tensorflow as tf
import os
import pandas as pd

DATASET_DIR='Stanford_Online_Products/'
CLASS=True
TARGET_SHAPE=(128, 128)
SAVE_FILE='df_emb'
PATH_MODEL='Model/hard_model'

#Первый вариант скрипта: Считывание происходит по директории, где изображения могут быть вложены в файлы,
# если они вложены, то Class=True, если в папке просто изображения Class=False.
# Лучше чтобы в файле не было ничего кроме изображений или папок с изображениями

def get_infopath(DATASET_DIR, Class=True):
    w=os.walk(DATASET_DIR)
    w=pd.DataFrame(w)
    if Class==True:
        normal_view=pd.concat([w.iloc[1:][0],w.iloc[1:][2]],axis=1)
        normal_view.reset_index(level=None,inplace=True,drop=True)
        normal_view=pd.concat([pd.Series(w.iloc[0][1]),normal_view],axis=1,ignore_index=True)
        num_image=[]
        Image_name=pd.DataFrame([])
        for i in range (len(normal_view)):
            num_image+=[len(normal_view.iloc[i][2])]
            Image_name=pd.concat([Image_name,pd.Series(normal_view.iloc[i][2])],ignore_index=True)
        class_name=normal_view.iloc[:][0].repeat(num_image)
        class_name=class_name.reset_index(level=None)
        path=normal_view.iloc[:][1].repeat(num_image)
        path=path.reset_index(level=None,drop=True)
        #все функции быстрые кроме вот этой:
        image_path=(path+'/'+Image_name.T).T
        part_df=pd.concat([class_name,image_path],axis=1,ignore_index=True)
        part_df=part_df.rename(columns={0:'Class',1:'Name_Class',2:'Image_Path'})
    else:
        Image_name=pd.Series(w.iloc[0][2])
        Image_name = Image_name.sort_values()
        path=w.iloc[:][0].repeat(len(Image_name))
        path=path.reset_index(level=None,drop=True)
        part_df=path.add(Image_name.T).T
        part_df = pd.DataFrame(part_df).rename(columns={0:'Image_Path'})
    return part_df

part_df=get_infopath(DATASET_DIR, Class=CLASS)
print(part_df)
test_data=tf.keras.preprocessing.image_dataset_from_directory(DATASET_DIR,labels=None, label_mode=None,class_names=None,shuffle=False,image_size=(128, 128))
"""
from matplotlib import pyplot as plt
plt.figure(figsize=(100, 100))
for images in test_data:
    for i in range(20):
        ax = plt.subplot(6, 6, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.axis("off")
plt.show()"""
embedding_model = EmbeddingModel(target_shape=TARGET_SHAPE)
embedding_model.load_weights(PATH_MODEL)
embeddings=pd.DataFrame(embedding_model.predict(test_data))
df=embeddings.join(part_df)
df.columns = df.columns.astype (str)
df.to_parquet(SAVE_FILE,engine='fastparquet')
#pf=pd.read_parquet(SAVE_FILE,engine='fastparquet')
