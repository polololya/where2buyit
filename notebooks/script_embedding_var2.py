from HardTripletLoss import EmbeddingModel
import pandas as pd
import tensorflow as tf
DATASET_DIR='Stanford_Online_Products/'
FULL_DATASET_FILE = 'Ebay_info.txt'
TARGET_SHAPE=(128, 128)
SAVE_FILE='df_emb'
PATH_MODEL='Model/hard_model'

#Второй вариант скрипта: считывание происходит по текстовому файлу, который пробегается по всей директории

def preprocess_image(TARGET_SHAPE,filename: tf.Tensor):
    """
    Загрузка изображения, декодирование, перевод значений в числа с плавающей точкой, а также изменение размера
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, TARGET_SHAPE)
    return image
full_dataset = pd.read_csv(f'{DATASET_DIR}{FULL_DATASET_FILE}', sep=' ')
embeddings=pd.DataFrame([])
#n=len(full_dataset)
n=10
embedding_model = EmbeddingModel(target_shape=TARGET_SHAPE)
embedding_model.load_weights(PATH_MODEL)
for i in range (n):#работает очень медленно, но я не знаю как оптимизировать
    ret=tf.Variable(DATASET_DIR+'/'+full_dataset.iloc[i,-1],dtype=tf.string)
    im=preprocess_image(TARGET_SHAPE,ret)
    embeddings =embeddings.append(pd.DataFrame(embedding_model.predict(tf.expand_dims(im, axis=0))),ignore_index=True)
    #test_model. если бы была бы embedding_model, то было бы embedding_model.
df=embeddings.join(full_dataset.iloc[:n,-3:])
df.columns = df.columns.astype (str)
df.to_parquet(SAVE_FILE,engine='fastparquet')
#pf=pd.read_parquet(SAVE_FILE,engine='fastparquet')


