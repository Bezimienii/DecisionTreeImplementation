import os
import pandas
import numpy

number_of_test_data = 40

def load_data():
    # load data into one csv
    fulldata = pandas.DataFrame()
    for file in os.listdir("DataForTree"):
        csv = pandas.read_csv("DataForTree/" + file)
        fulldata = pandas.concat([fulldata, csv]).reset_index(drop=True)

    fulldata = fulldata[["danceability", "energy", "key", "loudness", "mode", "speechiness",
                         "acousticness", "instrumentalness", "liveness", "valence", "tempo", "recommended"]]

    choice = numpy.random.choice(len(fulldata.index), number_of_test_data, replace=False)
    test_data_X, test_data_Y = fulldata.iloc[choice, :-1].to_numpy(), fulldata.loc[choice, ["recommended"]].to_numpy()
    dropped_rows = choice.tolist()
    train_data = fulldata.drop(labels=dropped_rows, axis=0)
    train_data_X, train_data_Y = train_data.iloc[:, 0:-1].to_numpy(), train_data[["recommended"]].to_numpy()
    return train_data_X, train_data_Y, test_data_X, test_data_Y
