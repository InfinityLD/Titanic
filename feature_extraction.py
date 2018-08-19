# coding: utf-8

import numpy as np
import pandas as pd

df_train = pd.read_csv('dataset/train.csv')
df_test = pd.read_csv('dataset/test.csv')


def feature_extraction(df_train):
    # name

    df_train['is_mr'] = [int('Mr.' in name) for name in df_train.Name]
    df_train['is_mrs'] = [int('Mrs.' in name) for name in df_train.Name]
    df_train['is_miss'] = [int('Miss.' in name) for name in df_train.Name]
    df_train['is_master'] = [int('Master.' in name) for name in df_train.Name]
    df_train['is_dr'] = [int('Dr.' in name) for name in df_train.Name]
    df_train['name_length'] = [int(len(name)) for name in df_train.Name]
    df_train['name_title_length'] = [int(len(name.split(' '))) for name in df_train.Name]

    # age

    def aging():
        age = []
        for i in range(df_train.shape[0]):
            info = df_train.iloc[i]
            if pd.isnull(info.Age) and (info.is_mr or info.is_mrs):
                age.append(33.0)
            elif pd.isnull(info.Age) and info.is_miss:
                age.append(22.0)
            elif pd.isnull(info.Age) and info.is_master:
                age.append(5.0)
            elif pd.isnull(info.Age) and info.is_dr:
                age.append(42.0)
            else:
                age.append(info.Age)
        return age

    df_train['Age'] = aging()

    # sex, cabin, embarked

    df_train['is_male'] = [int(sex == 'male') for sex in df_train.Sex]
    df_train['is_female'] = [int(sex == 'female') for sex in df_train.Sex]

    def cabin_feature():
        letter, num, lens = [], [], []

        for cabin in df_train.Cabin.fillna('0'):
            letter.append(str(cabin)[0])
            if cabin == '0':
                lens.append(0)
            else:
                lens.append(len(str(cabin)))
        return letter, lens

    def letter2cat(letter):
        letter_dict = {'0': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}

        return letter_dict[letter]

    df_train['Cabin'] = df_train['Cabin'].fillna(0)
    df_train['cabin_letter'], df_train['cabin_length'] = cabin_feature()
    df_train['cabin_cat'] = df_train['cabin_letter'].apply(letter2cat)

    def embark2cat(emb):
        emb_dict = {'S': 0, 'Q': 1, 'C': 2}
        return emb_dict[emb]

    df_train['Embarked'] = df_train['Embarked'].fillna('S')
    df_train['embarked'] = df_train['Embarked'].apply(embark2cat)

    return df_train


train = feature_extraction(df_train)
test = feature_extraction(df_test)

train = train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'cabin_letter'], axis=1)
test = test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'cabin_letter'], axis=1)

print(train.shape, test.shape)

train.to_csv('data/train.csv', index=None)
test.to_csv('data/test.csv', index=None)

