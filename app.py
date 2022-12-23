import base64

import numpy as np
import seaborn as sns
import pandas as pd
from flask import Flask, render_template, request, flash, redirect, url_for
import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM
import tensorflow as tf
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error

app = Flask(__name__)
app.config['export FLASK_ENV'] = 'development'
path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, 'Dataset/uploaded')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'csv', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

df = pd.read_csv('data/household_power_consumption.txt', sep=';',
                 parse_dates={'dt': ['Date', 'Time']}, infer_datetime_format=True,
                 low_memory=False, na_values=['nan', '?'], index_col='dt')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


"""
home page for the system
"""


@app.route('/')
@app.route('/home')
def hello_world():  # put application's code here
    return render_template('../../PowerConsumptionOnHouseHold/templates/Home.html')


@app.route('/Visualizations')
def Visualization():  # put application's code here
    return render_template('../../PowerConsumptionOnHouseHold/templates/visualisation.html')


@app.route('/hour_vis', methods=['GET', 'POST'])
def hour_visualization():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    img = BytesIO()
    i = 1
    cols = [0, 1, 3, 4, 5, 6]
    filename1 = "hourly.png"
    for col in cols:
        plt.subplot(len(cols), 1, i)
        plt.plot(df.resample('H').mean().values[:, col])
        plt.title(df.columns[col] + ' data resample over hour for mean', y=0.75, loc='left')
        if not (filename1 == ''):
            plt.savefig(filename1)
        i += 1
    plt.legend(title='labels', loc='upper left')
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return render_template('../../PowerConsumptionOnHouseHold/templates/hour_visualization.html', plot_url=plot_url)


@app.route('/daily_v', methods=['GET', 'POST'])
def daily_visual():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    img = BytesIO()
    i = 1
    cols = [0, 1, 3, 4, 5, 6]
    filename2 = "daily.png"
    for col in cols:
        plt.subplot(len(cols), 1, i)
        plt.plot(df.resample('D').mean().values[:, col])
        plt.title(df.columns[col] + ' data resample over day for mean', y=0.75, loc='center')
        if not (filename2 == ''):
            plt.savefig(filename2)
        i += 1
    plt.legend(title='labels', loc='upper left')
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return render_template('../../PowerConsumptionOnHouseHold/templates/Daily_visualisation.html', plot_url=plot_url)


@app.route('/monthly_v', methods=['GET', 'POST'])
def monthly_visual():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    img = BytesIO()
    i = 1
    cols = [0, 1, 3, 4, 5, 6]
    filename = "monthly.png"
    for col in cols:
        plt.subplot(len(cols), 1, i)
        plt.plot(df.resample('M').mean().values[:, col])
        plt.title(df.columns[col] + ' data resample over month for mean', y=0.75, loc='left')
        if not (filename == ''):
            plt.savefig(filename)
        i += 1
    plt.legend(title='labels', loc='upper left')
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return render_template('../../PowerConsumptionOnHouseHold/templates/Monthly_visualisation.html', plot_url=plot_url)


@app.route('/correlation_H', methods=['GET', 'POST'])
def correlation_hour():
    img = BytesIO()
    f = plt.figure(figsize=(14, 14))
    dfh = df.resample('H').mean()
    sns.heatmap(dfh.corr(), vmin=-1, vmax=1, annot=True)
    plt.title('Hourly resampling', size=12)
    plt.legend(title='labels', loc='upper left')
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return render_template('../../PowerConsumptionOnHouseHold/templates/hourly_correlation.html', plot_url=plot_url)


@app.route('/correlation_D', methods=['GET', 'POST'])
def correlation_day():
    img = BytesIO()
    f = plt.figure(figsize=(14, 14))
    dfd = df.resample('D').mean()
    sns.heatmap(dfd.corr(), vmin=-1, vmax=1, annot=True)
    plt.title('Daily resampling', size=12)
    plt.legend(title='labels', loc='upper left')
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return render_template('../../PowerConsumptionOnHouseHold/templates/daily_correlation.html', plot_url=plot_url)


@app.route('/correlation_M', methods=['GET', 'POST'])
def correlation_month():
    img = BytesIO()
    f = plt.figure(figsize=(14, 14))
    dfm = df.resample('M').mean()
    sns.heatmap(dfm.corr(), vmin=-1, vmax=1, annot=True)
    plt.title('Monthly resampling', size=12)
    plt.legend(title='labels', loc='upper left')
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return render_template('../../PowerConsumptionOnHouseHold/templates/monthly_correlation.html', plot_url=plot_url)


df_resample = df.resample('h').mean()
print(df_resample.shape)


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    dff = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(dff.shift(-i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(dff.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1)) for j in range(n_vars)]
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg


values = df_resample.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, 1, 1)
r = list(range(df_resample.shape[1] + 1, 2 * df_resample.shape[1]))
reframed.drop(reframed.columns[r], axis=1, inplace=True)
reframed.head()

# Data spliting into train and test data series. Only 4000 first data points are selected for traing purpose.
values = reframed.values
n_train_time = 4000
train = values[:n_train_time, :]
test = values[n_train_time:, :]
train_x, train_y = train[:, :-1], train[:, -1]
test_x, test_y = test[:, :-1], test[:, -1]
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

model = Sequential()
model.add(LSTM(100, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
                                           name='Adam'), loss='mean_squared_error')


# loss='mean_squared_error', optimizer='adam'
# Network fitting


@app.route('/training', methods=['POST', 'GET'])
def training_LSTM():
    if request.method == "POST":
        batch = request.form.get("batch_size")
        epochs = request.form.get("Epochs")
        history = model.fit(train_x, train_y, epochs=int(epochs), batch_size=int(batch),
                            validation_data=(test_x, test_y),
                            verbose=2,
                            shuffle=False)

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
    img = BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return render_template('../../PowerConsumptionOnHouseHold/templates/training.html', plot_url=plot_url)


@app.route('/prdediction', methods=['GET', 'POST'])
def prediction_LSTM():
    return render_template('../../PowerConsumptionOnHouseHold/templates/prediction.html')


@app.route('/prdediction_500', methods=['GET', 'POST'])
def prediction_500():
    values = reframed.values
    n_train_time = 4000
    train = values[:n_train_time, :]
    test = values[n_train_time:, :]
    train_x, train_y = train[:, :-1], train[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
    size = df_resample.shape[1]

    # Prediction test
    ypred = model.predict(test_x)
    test_x = test_x.reshape((test_x.shape[0], size))

    # invert scaling for prediction
    yscal = np.concatenate((ypred, test_x[:, 1 - size:]), axis=1)
    yscal = scaler.inverse_transform(yscal)
    yscal = yscal[:, 0]

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    Y_actual = np.concatenate((test_y, test_x[:, 1 - size:]), axis=1)
    Y_actual = scaler.inverse_transform(Y_actual)
    Y_actual = Y_actual[:, 0]
    rmse = np.sqrt(mean_squared_error(Y_actual, yscal))
    print('Test RMSE: %.3f' % rmse)
    print(yscal[1])
    img = BytesIO()
    aa = [x for x in range(500)]
    plt.figure(figsize=(10, 10))
    plt.plot(aa, Y_actual[:500], marker='.', label="actual")
    plt.plot(aa, yscal[:500], 'r', label="prediction")
    plt.ylabel(df.columns[0], size=15)
    plt.xlabel('Time step for first 500 hours', size=15)
    plt.legend(fontsize=15)
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return render_template('../../PowerConsumptionOnHouseHold/templates/prediction_500h.html', plot_url=plot_url)


@app.route('/prdediction_100', methods=['GET', 'POST'])
def prediction_1000():
    values = reframed.values
    n_train_time = 4000
    train = values[:n_train_time, :]
    test = values[n_train_time:, :]
    train_x, train_y = train[:, :-1], train[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
    size = df_resample.shape[1]

    # Prediction test
    y_pred = model.predict(test_x)
    test_x = test_x.reshape((test_x.shape[0], size))

    # invert scaling for prediction
    y_scaler = np.concatenate((y_pred, test_x[:, 1 - size:]), axis=1)
    y_scaler = scaler.inverse_transform(y_scaler)
    y_scaler = y_scaler[:, 0]

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    y_actual = np.concatenate((test_y, test_x[:, 1 - size:]), axis=1)
    y_actual = scaler.inverse_transform(y_actual)
    y_actual = y_actual[:, 0]
    img = BytesIO()
    aa = [x for x in range(1000)]
    plt.figure(figsize=(10, 10))
    plt.plot(aa, y_actual[20000:21000], marker='.', label="actual")
    plt.plot(aa, y_scaler[20000:21000], 'r', label="prediction")
    plt.ylabel(df.columns[0], size=15)
    plt.xlabel('Time step for 1000 hours from 20,000 to 21,000', size=15)
    plt.legend(fontsize=15)
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return render_template('../../PowerConsumptionOnHouseHold/templates/prediction_1000h.html', plot_url=plot_url)


if __name__ == '__main__':
    from waitress import serve

    serve(app, host="0.0.0.0", port=8080)
