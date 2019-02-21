import numpy as np
import psycopg2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
import pandas as pd
import simpleaudio as sa
import librosa
from keras.models import model_from_json
   
def get_X_train(mfcc_val):
    
    conn_string = "host='localhost' dbname='Audio_ML' user='postgres' password='Payton45'"
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()
    cur.execute("SET search_path TO Audio_ML;")
    cur.execute('SELECT "Mfcc_' +mfcc_val+ '" FROM "Audio_ML".training;')
    fetched = cur.fetchall()
    
    tuple_list = []
    for f in fetched:
        tuple_list.append(f[0])
        
    X = np.array(tuple_list)
    
    cur.close()
    conn.close()
    
    return X
    
def get_y_train():
    conn_string = "host='localhost' dbname='Audio_ML' user='postgres' password='Payton45'"
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()
    cur.execute("SET search_path TO Audio_ML;")
    
    cur.execute('SELECT "sound_class" FROM "Audio_ML".training;')
    fetched = cur.fetchall()
    
    tuple_list = []
    for f in fetched:
        tuple_list.append([f[0]])
        
    y = np.array(tuple_list)
    
    cur.close()
    conn.close()

    lenc = LabelEncoder()
    y = lenc.fit_transform(y)
    
    y = y.reshape(-1,1)
    
    enc = OneHotEncoder()
    y = enc.fit_transform(y)
    
    y = y.toarray()
    
    return y, lenc, enc
    
def recover_original_labels(y_in, enc, lenc):
    maxes = np.amax(y_in, axis=1)
    maxes = np.matlib.repmat(maxes.reshape(-1,1),1,10)
    y = (y_in >= maxes)*1
    y = sparse.csr_matrix(y)
    lenced = np.array([enc.active_features_[col] for col in y.sorted_indices().indices]) - enc.feature_indices_[:-1]
    return lenc.inverse_transform(lenced)
    
def get_X_test(mfcc_val):

    conn_string = "host='localhost' dbname='Audio_ML' user='postgres' password='Payton45'"
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()
    cur.execute("SET search_path TO Audio_ML;")
    cur.execute('SELECT "Mfcc_' +mfcc_val+ '" FROM "Audio_ML".testing;')
    fetched = cur.fetchall()
    
    tuple_list = []
    for f in fetched:
        tuple_list.append(f[0])
        
    X = np.array(tuple_list)
    
    cur.close()
    conn.close()
    
    return X 
    
def play_ID(ID):
    train = pd.read_csv('train/train.csv')
    test = pd.read_csv('test/test.csv')
    
    for index, sample in train.iterrows():
        if sample[0] == ID:
            play_sound('train/Train/'+str(ID)+'.wav')
            
    
    for index, sample in test.iterrows():
        if sample[0] == ID:
            play_sound('test/Test/'+str(ID)+'.wav')
            
            
            
def play_sound(file_str):
    audio, sample_rate = librosa.load(file_str, res_type='kaiser_fast')
    audio *= 32767 / np.max(np.abs(audio))
    audio = audio.astype(np.int16)
    play_obj = sa.play_buffer(audio, 1, 2, sample_rate)
    play_obj.wait_done()
    
    
def load_pretrained_cnn_model(model_id):
    CNN_file = open('models/'+model_id+'.json')
    CNN_str = CNN_file .read()
    CNN_file.close()
    CNN = model_from_json(CNN_str)
    CNN.load_weights('models/'+model_id+'.h5')
    return CNN
    
def save_cnn(CNN, model_id):
    CNN_json = CNN.to_json()
    with open('models/'+model_id+'.json', "w") as CNN_file:
        CNN_file.write(CNN_json)
    CNN.save_weights('models/'+model_id+'.h5')
    
    
    
    
    
    
    