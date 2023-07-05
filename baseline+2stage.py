import os
import random

import lightgbm as lgb
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from loguru import logger

from config import *
from features import *

random.seed(605)
np.random.seed(605)


def process(f: str, cols=None):
    """_summary_

    Args:
        f (str): _description_
        cols (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    df = pd.read_csv(f, sep="\t").fillna(0)
    for col, v in process_dict.items():
        df[col] = (df[col] / v).round()

    df["date_mod"] = df["f_1"] % 7
    df["f_4_6"] = merge_cols(df[["f_4", "f_6"]], "./data/f_4_6_le")

    df["date_mod_f_11"] = merge_cols(
        df[["date_mod", "f_11"]], "./data/date_mod_f_11_le")

    df = merge(df, 'f_11', 'f_71')
    df = merge(df, 'f_13', 'f_74')

    return df


def prepare_dataset(data_func):
    logger.info("loading training set.")

    data = pd.concat(
        Parallel(n_jobs=30)(
            delayed(process)(train_data_path / f, cols=None)
            for f in sorted(os.listdir(train_data_path))
        )
    ).reset_index(drop=True)
    cond = data["f_1"] == 66

    logger.info("loading test set.")
    test_data = process(test_data_path / "000000000000.csv",
                        cols=None)

    logger.info("start beta target encoder.")

    data = data_func(data)

    # beta target encoder
    N_min = 1000
    feature_cols = []
    cat_cols = ["f_4", "f_6", "f_13", "f_15"]
    # encode variables
    for c in cat_cols:
        # fit encoder
        be = BetaEncoder(c)
        be.fit(data, "is_installed")

        # mean
        feature_name = f"{c}_mean"
        data[feature_name] = be.transform(data, "mean", N_min)
        test_data[feature_name] = be.transform(test_data, "mean", N_min)
        feature_cols.append(feature_name)

        # mode
        feature_name = f"{c}_mode"
        data[feature_name] = be.transform(data, "mode", N_min)
        test_data[feature_name] = be.transform(test_data, "mode", N_min)
        feature_cols.append(feature_name)

        # median
        feature_name = f"{c}_median"
        data[feature_name] = be.transform(data, "median", N_min)
        test_data[feature_name] = be.transform(test_data, "median", N_min)
        feature_cols.append(feature_name)

        # var
        feature_name = f"{c}_var"
        data[feature_name] = be.transform(data, "var", N_min)
        test_data[feature_name] = be.transform(test_data, "var", N_min)
        feature_cols.append(feature_name)

        # skewness
        feature_name = f"{c}_skewness"
        data[feature_name] = be.transform(data, "skewness", N_min)
        test_data[feature_name] = be.transform(test_data, "skewness", N_min)
        feature_cols.append(feature_name)

        # kurtosis
        feature_name = f"{c}_kurtosis"
        data[feature_name] = be.transform(data, "kurtosis", N_min)
        test_data[feature_name] = be.transform(test_data, "kurtosis", N_min)
        feature_cols.append(feature_name)

    df = pd.concat([data, test_data], axis=0)

    df = merge(df, 'f_2', 'f_42')
    df = merge(df, 'f_4', 'f_54')
    df = merge(df, 'f_6', 'f_68')
    df = merge(df, 'f_8', 'f_71')

    data, test_data = df.loc[df['f_1'] != 67], df.loc[df['f_1'] == 67]
    test_data = test_data.drop(["is_clicked", "is_installed"], axis=1)
    train_data, dev_data = data[~cond], data[cond]

    train_x, train_y, dev_x, dev_y = (
        train_data.drop(drop_cols + ["is_clicked", "is_installed"], axis=1),
        train_data["is_installed"],
        dev_data.drop(drop_cols + ["is_clicked", "is_installed"], axis=1),
        dev_data["is_installed"],
    )

    return train_x, train_y, dev_x, dev_y, test_data


def train_lgb(train_x, train_y, dev_x, dev_y):
    """_summary_

    Args:
        train_x (_type_): _description_
        train_y (_type_): _description_
        dev_x (_type_): _description_
        dev_y (_type_): _description_

    Returns:
        _type_: _description_
    """
    train_dataset = lgb.Dataset(train_x, label=train_y)
    dev_dataset = lgb.Dataset(dev_x, label=dev_y)

    logger.info("start training.")
    model = lgb.train(
        model_params,
        train_dataset,
        valid_sets=[dev_dataset],
        verbose_eval=50,
    )

    return model


def test(model, out_file, test_data, drop_cols):
    """_summary_

    Args:
        model (_type_): _description_
        out_file (_type_): _description_
        test_data (_type_): _description_
        drop_cols (_type_): _description_
    """
    probs = model.predict(test_data.drop(drop_cols, axis=1))
    submission = pd.DataFrame(
        {"RowId": test_data["f_0"], "is_clicked": 0, "is_installed": probs}
    )
    submission.to_csv(out_file, sep="\t", index=False, header=True)


def main():
    # stage 1
    def data_func1(data: pd.DataFrame):
        def func(x, y):
            if (x+y > 0):
                return 1
            else:
                return 0
        data['is_clicked'] = data.apply(lambda x: func(
            x["is_clicked"], x['is_installed']), axis=1)
        return data

    train_x, train_y, dev_x, dev_y, test_data = prepare_dataset(data_func1)
    model = train_lgb(train_x, train_y, dev_x, dev_y)
    test(model, "base_6.01_stage1.txt", test_data, drop_cols)

    # stage 2

    def data_func2(data: pd.DataFrame):
        return data.loc[(data['is_clicked'] == 1) | (data['is_installed'] == 1)]

    train_x, train_y, dev_x, dev_y, test_data = prepare_dataset(data_func2)
    model = train_lgb(train_x, train_y, dev_x, dev_y)
    test(model, "base_6.01_stage2.txt", test_data, drop_cols)

    # merge two results

    prefer_df = pd.read_csv("./base_6.01_stage1.txt", sep="\t")
    install_df = pd.read_csv("./base_6.01_stage2.txt", sep="\t")

    install_df['is_installed'] = prefer_df['is_installed'] * \
        install_df['is_installed']
    install_df.to_csv("./baseline_prefer_install_ctcvr.txt",
                      sep='\t', index=False, header=True)


if __name__ == "__main__":
    logger.info("start.")
    main()
    logger.info("end.")
