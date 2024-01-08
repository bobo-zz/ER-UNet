import tensorflow as tf
tf.config.list_physical_devices("GPU")
import tensorflow.keras as keras
import utils
import dataset
from metric import *
import numpy as np
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import TensorBoard,ReduceLROnPlateau
import lr
import er_unet, unet
from tfrecored import read_tfrecord
from loss import weighted_mae_loss2


train_path = r'D://7_3dataset//step14//train'
test_path = r'D://7_3dataset//step14//test'

batch= 32

train_data = read_tfrecord(tfrecord_path= r'D:\7_3dataset\step14\train.tfrecord', batch_size=batch, original_path=train_path)
test_data = read_tfrecord(tfrecord_path= r'D:\7_3dataset\step14\test.tfrecord', batch_size=batch, original_path=test_path)
## 设置 optimitre 和 loss
model = er_unet.create_model()
#model = unet2.create_model()
#optimizer = tf.optimizers.SGD( learning_rate = 0.001, momentum = 0.99)
optimizer = tf.optimizers.Adam(learning_rate=0.0005)

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
    return lr
lr_metric = get_lr_metric(optimizer)
model.compile(optimizer = optimizer, loss = weighted_mae_loss2, metrics=['mse', 'mae', lr_metric])
model.summary()

#### 设置 学习率变化策略
###### ----------------------------
# Number of training samples.
sample_count = 5910
# Total epochs to train.
epochs = 200
# Number of warmup epochs.
warmup_epoch = 10  
# Training batch size, set small value here for demonstration purpose.
batch_size = batch
# Base learning rate after warmup.
learning_rate_base = 0.0005   
total_steps = int(epochs * sample_count / batch_size)
# Compute the number of warmup batches.
warmup_steps = int(warmup_epoch * sample_count / batch_size)
# Compute the number of warmup batches.
warmup_batches = warmup_epoch * sample_count / batch_size
# Create the Learning rate scheduler.
warm_up_lr = lr.WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                        total_steps=total_steps,
                                        warmup_learning_rate=0.0,
                                        warmup_steps=warmup_steps,
                                        hold_base_rate_steps=0)
###### ------------------------------

### 设置优化器
#Tensorboard = TensorBoard(log_dir=r"D:\Unet\result\model\7thall_batch8_s32_res_warm_leakrelu_all_01_adam_dataset_se", histogram_freq=1, write_grads=True,write_images=True,profile_batch=16)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min',factor=0.7,
                              patience=4, min_lr=0.0000000001)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, mode='min')
csv_logger = tf.keras.callbacks.CSVLogger(filename=r'D:\Unet\csv_file\Model1_UNet_freq' + '_train.csv', separator=',',
                                          append=False)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "./result/Model1_UNet_freq.h5", save_best_only=True
)

history = model.fit_generator(
    train_data,
    validation_data=test_data,
    epochs=200,
    verbose=2,
    callbacks=[early_stopping_cb,checkpoint_cb, csv_logger,warm_up_lr],
    workers=8,
)


## 加载数据 ,测试结果
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
    return lr
optimizer = tf.optimizers.Adam(learning_rate=0.0005)  

lr_metric = get_lr_metric(optimizer)
def my_leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.01)  

file_path = r'D:\newdata\step14\test'
file_list = os.listdir(file_path)
file_path = r"D:\Unet\result\ER_UNet_VIS.h5"
model = load_model(filepath=file_path, custom_objects={'weighted_mae_loss2':weighted_mae_loss2,'lr':lr_metric,  'HaarWaveLayer2D': HaarWaveLayer2D,'my_leaky_relu':my_leaky_relu})
# data_channel = [2,5,6,7,12,13,15]
data_channel = [2,5,6]  #设置选择可见光及近红外通道，如果是输入IR则是 7 12  13 15
batch_data = []
y_obs = []
time_list = []
for i in range(len(file_list)):
    data = np.load(os.path.join(file_path, file_list[i]))
    y_obs.append(data[0])
    time_list.append(str(file_list[i].split('_')[0]))
    batch_x = np.ones(shape=[len(data_channel), 416, 448], dtype='float32')
    for j in range(len(data_channel)):
        batch_x[j] = data[data_channel[j]]
    batch_data.append(batch_x)
y_obs = np.array(y_obs)
time_list = np.array(time_list)
batch_data = np.array(batch_data)
batch_data = np.transpose(batch_data, axes=(0, 2, 3, 1))

y_pre = model.predict(batch_data)
y_pre = np.squeeze(y_pre)
y_obs = y_obs * 60
y_pre = y_pre * 60
np.save('unet24——y_pre.npy', y_pre)
np.save('unet24——y-obs.npy', y_obs)
#np.save('day-unet——y_pre.npy', y_pre)
#np.save('day-unet——y-obs.npy', y_obs)
print('R2:',R2(y_obs,y_pre))
print('MAE:', MAE(y_obs,y_pre))
print('RMSE:', RMSE(y_obs,y_pre))
print('TS:15dBZ', TS(y_obs,y_pre,15))
print('TS:25dBZ', TS(y_obs,y_pre,25))
print('TS:35dBZ', TS(y_obs,y_pre,35))
print('TS:45dBZ', TS(y_obs,y_pre,45))
print('FAR:15dBZ', FAR(y_obs,y_pre,15))
print('FAR:25dBZ', FAR(y_obs,y_pre,25))
print('FAR:35dBZ', FAR(y_obs,y_pre,35))
print('FAR:45dBZ', FAR(y_obs,y_pre,45))
print('POD:15dBZ', POD(y_obs,y_pre,15))
print('POD:25dBZ', POD(y_obs,y_pre,25))
print('POD:35dBZ', POD(y_obs,y_pre,35))
print('POD:45dBZ', POD(y_obs,y_pre,45))