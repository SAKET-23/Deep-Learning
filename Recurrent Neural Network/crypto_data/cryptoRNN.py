import pandas as pd
from collections import deque
import random
import numpy as np
from sklearn import preprocessing
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM,BatchNormalization
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard , ModelCheckpoint
from tensorflow.compat.v1.keras.layers import CuDNNLSTM



SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT  ="LTC-USD"
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{SEQ_LEN}-SEQ -{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

def classify(current , future):
    if float(future) > float(current):
        return 1
    else:
        return 0


main_df = pd.DataFrame() # begin empty
ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]  # the 4 ratios we want to consider
for ratio in ratios:  # begin iteration

    dataset = "crypto_data/{}.csv".format(ratio)  # get the full path to the file.
    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])  # read in specific file

    # rename volume and close to include the ticker so we can still which close/volume is which:
    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)

    df.set_index("time", inplace=True)  # set time as index so we can join them on this shared time
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]  # ignore the other columns besides price and volume
    # print(df.head())

    if len(main_df)==0:  # if the dataframe is empty
        main_df = df  # then it's just the current df
    else:  # otherwise, join this data to the main one
        main_df = main_df.join(df)


main_df["future"]  =main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)
main_df["target"] = list(map(classify ,main_df[f"{RATIO_TO_PREDICT}_close"], main_df["future"] ))

# print(main_df[[f"{RATIO_TO_PREDICT}_close","future","target"]].head())

times = sorted(main_df.index.values)
last_5pct = times[-int(0.05 * len(times))]

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]


def preprocess(df):
    df = df.drop("future" , 1)
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace = True)
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace = True)

    sequential_data =  []
    prev_days = deque(maxlen = SEQ_LEN)
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)
    buys = []
    sells = []
    for seq,target in sequential_data:
        if target == 0:
            sells.append([seq , target])
        elif target == 1:
            buys.append([seq , target])

    random.shuffle(buys)
    random.shuffle(sells)
    lower = min(len(buys) , len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells
    random.shuffle(sequential_data)
    X = []
    y = []

    for seq,target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X) , np.array(y)

# preprocess(main_df)
train_x , train_y = preprocess(main_df)
test_x , test_y = preprocess(validation_main_df)

trian_x = tf.keras.utils.normalize(train_x ,axis = 1)
test_x= tf.keras.utils.normalize(test_x ,axis = 1)

model = Sequential()
model.add(CuDNNLSTM(128, input_shape = (train_x.shape[1:]) , return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128 , return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32 , activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(1 , activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001 , decay = 1e-6)

model.compile(loss = "sparse_categorical_crossentropy" , optimizer=opt , metrics = ["accuracy"])

tensorBoard = TensorBoard(log_dir= f'logs/{NAME}')
#
filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(test_x,test_y),
    callbacks=[tensorBoard],
)
score = model.evaluate(test_x, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
