import helper as h
import pandas as pd

y, lenc, enc = h.get_y_train()

X_test_30 = h.get_X_test('30')
X_test_40 = h.get_X_test('40')
X_test_50 = h.get_X_test('50')
X_test_60 = h.get_X_test('60')
X_test_70 = h.get_X_test('70')
X_test_80 = h.get_X_test('80')
X_test_90 = h.get_X_test('90')

CNN_30 = h.load_pretrained_cnn_model('mfcc_30')
CNN_40 = h.load_pretrained_cnn_model('mfcc_40')
CNN_50 = h.load_pretrained_cnn_model('mfcc_50')
CNN_60 = h.load_pretrained_cnn_model('mfcc_60')
CNN_70 = h.load_pretrained_cnn_model('mfcc_70')
CNN_80 = h.load_pretrained_cnn_model('mfcc_80')
CNN_90 = h.load_pretrained_cnn_model('mfcc_90')

pred_30 = CNN_30.predict(X_test_30)
pred_40 = CNN_40.predict(X_test_40)
pred_50 = CNN_50.predict(X_test_50)
pred_60 = CNN_60.predict(X_test_60)
pred_70 = CNN_70.predict(X_test_70)
pred_80 = CNN_80.predict(X_test_80)
pred_90 = CNN_90.predict(X_test_90)

pred = pred_30 + pred_50 + pred_60 + pred_80 + pred_90 # pred_40 + pred_70 

predictions_labeled = h.recover_original_labels(pred, enc, lenc)

test = pd.read_csv('test/test.csv')
test['Class'] = predictions_labeled
test.to_csv('sub_ensemble_5689.csv', index=False)