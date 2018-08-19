# Titanic

dataset from kaggle: [Titanic](https://www.kaggle.com/c/titanic/)

dataset contains columns:

```
'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
```

after feature extraction:

```
'PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
       'is_mr', 'is_mrs', 'is_miss', 'is_master', 'is_dr', 'name_length',
       'name_title_length', 'is_male', 'is_female', 'cabin_length',
       'cabin_cat', 'embarked'
```

__with an acc: 0.76555__

### feature_extraction.py



__feature_extraction.py__ is used to extracting features in dataset：

According to __Name__, __Age__, __Sex__, __Cabin__, __Embarked__

+ Name (non-null)

  + 'is_mr': people who has 'Mr.' in name
  + 'is_mrs': people who has 'Mrs.' in name
  + 'is_miss': people who has 'Miss.' in name
  + 'is_master': people who has 'Master' in name
  + 'is_dr': people who has 'Dr.' in name
  + 'name_length': longer name_length may be a noble,  more likely to be rescued
  + 'name_title_length':  longer title may be a noble,  more likely to be rescued

+ Age (with null)

  + fill the Age NAs with Name features like 'is_mr', 'is_mrs', using mean number
    + Mr. avg age: 32
    + Mrs. avg age: 35
    + Miss. avg age: 22
    + Master. avg age: 4.5
    + Dr. avg age: 42

+ Sex (non-null)

  + 'is_male', 'is_female'

+ Cabin (with null)

  constructed like 'A34' (single value), 'C23 C25 C27' (multiple values)

  + fill na with '0'
  + 'cabin letter':  first letter in cabins
  + 'lens': those who has ordered more cabins have more in 'lens', possibly rich
  + 'cabin_cat': convert cabin letters to category feature using dict

+ Embarked (with null)

  like 'S', 'C', 'Q'

  + fill na with 'S'
  + 'embarked': convert embarked letters to category feature using dict