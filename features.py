import numpy as np
import pandas as pd
from functional import seq
import pickle
from loguru import logger


def merge_cols(df: pd.DataFrame, encoder_f=None):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        encoder_f (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    logger.info(f"merging columns: {df.columns}")
    encoder = pickle.load(open(encoder_f, "rb")) if encoder_f else None
    return (
        seq(df.values.astype(str).tolist())
        .map(lambda x: "@".join(x))
        .map(lambda x: encoder.get(x, -1) if encoder else x)
    ).list()


def merge(df: pd.DataFrame, col_a: str, col_b: str):
    """无序/有序类别/数值特征的组合特征


    Args:
        df (pd.DataFrame): 输入数据集
        col_a (str): 类别型特征
        col_b (str): 数值型特征

    Returns:
        _type_: _description_
    """
    df[col_a + "_" + col_b + "_mean"] = (
        df.groupby(col_a)[col_b].transform("mean").values
    )
    df[col_a + "_" + col_b + "_median"] = (
        df.groupby(col_a)[col_b].transform("median").values
    )
    df[col_a + "_" + col_b +
        "_std"] = df.groupby(col_a)[col_b].transform("std").values
    df[col_a + "_" + col_b +
        "_max"] = df.groupby(col_a)[col_b].transform("max").values
    df[col_a + "_" + col_b +
        "_min"] = df.groupby(col_a)[col_b].transform("min").values

    df[col_b + "_div_" + col_a + "_" + col_b + "_mean"] = df[col_b] / (
        df[col_a + "_" + col_b + "_mean"] + 1e-5
    )
    df[col_b + "_div_" + col_a + "_" + col_b + "_median"] = df[col_b] / (
        df[col_a + "_" + col_b + "_median"] + 1e-5
    )

    df[col_b + '_minus_' + col_a + "_" + col_b + "_mean"] = df[col_b] - \
        df[col_a + "_" + col_b + "_mean"]

    df[col_a + "_" + col_b +
        "_std"] = df.groupby(col_a)[col_b].transform("std").values

    df[col_b+'_minus_' + col_a + "_" + col_b + "_mean" + '_norm'] = (
        df[col_b] - df[col_a + "_" + col_b + "_mean"]
    ) / (df[col_a + "_" + col_b + "_std"] + 1e-9)

    df[col_a + "_" + col_b + '_mf1_1'] = df[col_a + "_" +
                                            col_b + "_median"] - df[col_a + "_" + col_b + "_mean"]
    df[col_a + "_" + col_b + '_mf1_2'] = df[col_a +
                                            "_" + col_b + '_mf1_1'].map(abs)

    df[col_a + "_" + col_b + '_mf2'] = df[col_a + "_" +
                                          col_b + "_median"] / df[col_a + "_" + col_b + "_mean"]

    df[col_a + "_" + col_b +
       "_cv"] = df[col_a + "_" + col_b +
                   "_std"] / df[col_a + "_" + col_b + "_mean"]

    return df


class BetaEncoder(object):
    def __init__(self, group):
        self.group = group
        self.stats = None

    # get counts from df
    def fit(self, df, target_col):
        self.prior_mean = np.mean(df[target_col])

        stats = df[[target_col, self.group]].groupby(self.group)
        # count和sum
        stats = stats.agg(["sum", "count"])[target_col]
        stats.rename(columns={"sum": "n", "count": "N"}, inplace=True)
        stats.reset_index(level=0, inplace=True)
        self.stats = stats

    # extract posterior statistics
    def transform(self, df, stat_type, N_min=1):
        df_stats = pd.merge(df[[self.group]], self.stats, how="left")
        n = df_stats["n"].copy()
        N = df_stats["N"].copy()

        # fill in missing
        nan_indexs = np.isnan(n)
        n[nan_indexs] = self.prior_mean
        N[nan_indexs] = 1.0

        # prior parameters
        N_prior = np.maximum(N_min - N, 0)
        alpha_prior = self.prior_mean * N_prior
        beta_prior = (1 - self.prior_mean) * N_prior

        # posterior parameters
        alpha = alpha_prior + n
        beta = beta_prior + N - n

        # calculate statistics
        if stat_type == "mean":
            num = alpha
            dem = alpha + beta

        elif stat_type == "mode":
            num = alpha - 1
            dem = alpha + beta - 2

        elif stat_type == "median":
            num = alpha - 1 / 3
            dem = alpha + beta - 2 / 3

        elif stat_type == "var":
            num = alpha * beta
            dem = (alpha + beta) ** 2 * (alpha + beta + 1)

        # elif stat_type == "skewness":
        #     num = 2 * (beta - alpha) * np.sqrt(alpha + beta + 1)
        #     dem = (alpha + beta + 2) * np.sqrt(alpha * beta)

        # elif stat_type == "kurtosis":
        #     num = 6 * (alpha - beta) ** 2 * (alpha + beta + 1) - alpha * beta * (
        #         alpha + beta + 2
        #     )
        #     dem = alpha * beta * (alpha + beta + 2) * (alpha + beta + 3)

        elif stat_type == "skewness":
            num = alpha - beta
            dem = np.sqrt(alpha * beta * (alpha + beta + 1))

        elif stat_type == "kurtosis":
            num = alpha * beta * (alpha + beta + 1)
            dem = (alpha + beta + 2) * (alpha + beta) * (alpha * beta)
            value = (num / dem) - 3
            value[np.isnan(value)] = np.nanmedian(value)
            return value

        value = num / dem
        value[np.isnan(value)] = np.nanmedian(value)
        return value
