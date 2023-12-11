# RecSys Challenge 2023
> **Predicting Conversion Rate in Advertising Systems: A Two-Stage Approach with LightGBM**\
> Lulu Wang, Yu Zhang, Huayang Zhao, Zhewei Song, Jiaxin Hu\
> Paper: https://github.com/colorblank/recsys-challenge-2023/blob/main/recsys2023_challenge.pdf

This project is a solution of Recsys Challenge 2023 provided by the team ```Ainvest```. For more details about this challenge, please visit the official website (https://sharechat.com/recsys2023).


***Rank***: 4th at Company Leardboard


## Environment Setup 
Please refer to `requirements.txt` for environment installation. 
Note: The best result is obtained using GPU version of LightGBM.

## Project Structure
```
recsys-challenge-2023
├── README.md
├── baseline+2stage.py
├── baseline.py
├── config.py
├── data
│   ├── date_mod_f_11_le
│   ├── f_4_6_le
│   ├── test
│   └── train
├── features.py
└── requirements.txt
```
## Steps to Reproduce
In this challenge, we first utilize LightGBM in conjunction with feature engineering to establish a strong baseline model. Then, we adopt a two-stage modeling approach to estimate the download probability. Finally, we employ an ensemble method to incorporate a deep learning model for enhanced diversity.

### Download Dataset 

```shell
# 1. download from official website
wget https://cdn.sharechat.com/2a161f8e_1679936280892_sc.zip
# 2. unzip file
unzip 2a161f8e_1679936280892_sc.zip
# 3. move files to target folder
mv sharechat_recsys2023_data/test/* data/test/
mv sharechat_recsys2023_data/train/* data/train/

# 4. remove zip and other files
rm -r sharechat_recsys2023_data/
rm -r __MACOSX/
rm 2a161f8e_1679936280892_sc.zip
```

The data folder: 
```
data/
├── date_mod_f_11_le
├── f_4_6_le
├── test
│   └── 000000000000.csv
└── train
    ├── 000000000000.csv
    ├── 000000000001.csv
    ├── 000000000002.csv
    |__ ...........
    ├── 000000000026.csv
    ├── 000000000027.csv
    ├── 000000000028.csv
    └── 000000000029.csv
```

### Train & Test

1. (Optional) train a baseline model with lightgbm: `python baseline.py`. The submission result obtain `6.01`.

2. train a two-stage model with lightgbm: `python baseline+2stage.py`. The submission result obtain `5.96`.

3. Calibration on result score `is_installed`:
$$
y' = \frac{y}{y + (1- y) / w},
$$
where $w$ set to 0.958.

Final Online Score: `5.949816`.


