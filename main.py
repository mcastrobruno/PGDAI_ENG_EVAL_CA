from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
from get_chained_variable import *
from set_dependent_variable import *
import random

seed = 0
random.seed(seed)
np.random.seed(seed)


def load_data():
    # load the input data
    df = get_input_data()
    return df


def preprocess_data(df):
    df = de_duplication(df)
    df = noise_remover(df)
    return df


def get_embeddings(df: pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df


def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)


def perform_modelling(data: Data, df: pd.DataFrame, name):
    model_predict(data, df, name)

def get_chained_variable(df:pd.DataFrame):
    chain_1 = group_df["y2"]
    chain_2 = group_df["y2"] + group_df["y3"]
    chain_3 = group_df["y2"] + group_df["y3"] + group_df["y4"]
    chained_variables = [chain_1, chain_2, chain_3]
    return chained_variables

if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    grouped_df = df.groupby(Config.GROUPED)

    for name, group_df in grouped_df:
        for chain in get_chained_variable(group_df):
            set_dependent_variable(chain, group_df)
            X, group_df = get_embeddings(group_df)
            data = get_data_object(X, group_df)
            perform_modelling(data, group_df, name)
