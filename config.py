from pathlib import Path


train_data_path = Path("./data/train")
test_data_path = Path("./data/test")


process_dict = {
    "f_42": 0.03856407,
    **{f"f_{i}": 0.57112147 for i in range(44, 51)},
    **{f"f_{i}": 0.03856407 for i in range(52, 58)},
    "f_60": 8.07946039,
    "f_61": 0.147850899,
    "f_62": 0.129299709,
    "f_63": 0.355221093,
    "f_71": 0.57112147,
    "f_72": 0.57112147,
    "f_73": 0.57112147,
    "f_74": 0.03856407,
    "f_75": 0.03856407,
    "f_76": 0.03856407,
    "f_77": 37.38457512,
    "f_78": 37.38457512,
    "f_79": 37.38457512,
}

cat_features = [f"f_{i}" for i in range(2, 33)]


model_params = {
    "objective": "binary",
    "metric": ['auc', "binary_logloss"],
    "boosting": "gbdt",
    "num_leaves": 31,
    "max_depth": -1,
    "num_iterations": 20000,
    "learning_rate": 0.01,
    "early_stopping_rounds": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "device": "GPU",
}

drop_cols = ["f_0", "f_7", "f_9", "f_23", "f_24", "f_47", "f_49"]
