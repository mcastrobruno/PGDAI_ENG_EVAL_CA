from pandas import DataFrame

from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
import random
import pandas as pd

seed = 0
random.seed(seed)
np.random.seed(seed)


def load_data() -> pd.DataFrame:  # str: # pd.DataFrame:  # CHANGE-> pd.DataFrame added
    """

    :rtype: object
    """
    # load the input data
    # df = get_input_data()
    # ORIGINAL-NOW CHANGED # df = get_input_data()
    df = pd.read_csv('C:\\Users\Arthur\PycharmProjects\Engineering_Evaluation_CA\data\AppGallery.csv')
    #df.drop(df.loc[:, -2:].columns, axis=1) # CHANGE # df.drop([11,12], axis=1)
    df = df.loc[:, ~df.columns.str.match("Unnamed: 11")] # CHANGE
    df = df.loc[:, ~df.columns.str.match("Unnamed: 12")] # CHANGE
    return df  # df.applymap(str) # str(df) CHANGE


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:  # CHANGE -> pd.DataFrame added
    """

    :type df: object
    """
    # De-duplicate input data
    #df = df.applymap(str) # CHANGE
    #for string in df:
        #df = de_duplication(df)
    #df = str(df) # CHANGE

    # CHANGE
    df["Mailbox"] = df["Mailbox"].astype(str)
    df["Ticket Summary"] = df["Ticket Summary"].astype(str)
    df["Interaction content"] = df["Interaction content"].astype(str)

    df = de_duplication(df)
    # remove noise in input data
    df = noise_remover(df)
    # translate data to english
    # df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df


def get_embeddings(df: pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df


def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)


def perform_modelling(data: Data, df: pd.DataFrame, name):
    model_predict(data, df, name)


if __name__ == '__main__':
    df: DataFrame = load_data() # CHANGE
    # df.applymap(str) # CHANGE
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    grouped_df = df.groupby(Config.GROUPED)
    for name, group_df in grouped_df:
        print(name)
        X, group_df = get_embeddings(group_df)
        data = get_data_object(X, group_df)
        perform_modelling(data, group_df, name)
