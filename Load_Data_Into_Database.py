import pandas as pd
import librosa
import numpy as np  
import psycopg2


def add_to_DB_test(ID_str, ID_num, cur):
    try:
        X, sample_rate = librosa.load(ID_str, res_type='kaiser_fast') 
        Mfcc_30 = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=30).T,axis=0) 
        Mfcc_40 = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
        Mfcc_50 = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T,axis=0) 
        Mfcc_60 = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=60).T,axis=0) 
        Mfcc_70 = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=70).T,axis=0) 
        Mfcc_80 = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=80).T,axis=0) 
        Mfcc_90 = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=90).T,axis=0) 
        insert_statement = 'INSERT INTO "Audio_ML".testing('+\
    	'"ID", "Mfcc_30", "Mfcc_40", "Mfcc_50", "Mfcc_60", "Mfcc_70", "Mfcc_80", "Mfcc_90")'+\
    	'VALUES ('+str(ID_num)+', '+make_sql_list(Mfcc_30)+', '+make_sql_list(Mfcc_40)+', '+\
         make_sql_list(Mfcc_50)+', '+make_sql_list(Mfcc_60)+', '+make_sql_list(Mfcc_70)+\
         ', '+make_sql_list(Mfcc_80)+', '+make_sql_list(Mfcc_90)+');'
        cur.execute(insert_statement)
        return 0
    except:
        print(ID_num, 'FAIL')
        return 1
    
    
def add_to_DB_train(ID_str, ID_num, class_str, cur):
    try:
        X, sample_rate = librosa.load(ID_str, res_type='kaiser_fast') 
        Mfcc_30 = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=30).T,axis=0) 
        Mfcc_40 = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
        Mfcc_50 = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T,axis=0) 
        Mfcc_60 = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=60).T,axis=0) 
        Mfcc_70 = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=70).T,axis=0) 
        Mfcc_80 = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=80).T,axis=0) 
        Mfcc_90 = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=90).T,axis=0) 
        insert_statement = 'INSERT INTO "Audio_ML".training('+\
    	'"ID", "Mfcc_30", "Mfcc_40", "Mfcc_50", "Mfcc_60", "Mfcc_70", "Mfcc_80", "Mfcc_90", sound_class) '+\
    	'VALUES ('+str(ID_num)+', '+make_sql_list(Mfcc_30)+', '+make_sql_list(Mfcc_40)+', '+\
         make_sql_list(Mfcc_50)+', '+make_sql_list(Mfcc_60)+', '+make_sql_list(Mfcc_70)+\
         ', '+make_sql_list(Mfcc_80)+', '+make_sql_list(Mfcc_90)+', '+"'"+ class_str +"'"+');'
        cur.execute(insert_statement)
        return 0
    except:
        print(ID_num, 'FAIL')
        return 1
        
    

def make_sql_list(arr):
    ret_str = 'ARRAY'+str(arr.tolist()).replace('\n','')
    return ret_str
    


if __name__ == "__main__":
    FAILS = 0
    
    conn_string = "host='localhost' dbname='Audio_ML' user='postgres' password='Payton45'"
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()
    cur.execute("SET search_path TO Audio_ML;")
    cur.execute('DELETE FROM "Audio_ML".testing')
    cur.execute('DELETE FROM "Audio_ML".training')
    
    train = pd.read_csv('train/train.csv')
    test = pd.read_csv('test/test.csv')

    for index, sample in train.iterrows():
        ID_str = 'train/Train/' + str(sample[0]) + '.wav'
        ID_num = sample[0]
        FAILS = FAILS + add_to_DB_train(ID_str, ID_num, sample[1], cur)
        print(sample[0])
        
    for index, sample in test.iterrows():
        ID_str = 'test/Test/' + str(sample[0]) + '.wav'
        ID_num = sample[0]
        FAILS = FAILS + add_to_DB_test(ID_str, ID_num, cur)
        print(sample[0])
        
    conn.commit()
    
    print('fails', FAILS)
    
    cur.close()
    conn.close()
        
        