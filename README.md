# PeaViner

Scientific project to efficiently find the most optimal short classifier.

More info will be presented later.

## Install
PeaViner can be installed from [GitHub](https://github.com/EgorDudyrev/PeaViner.git):

```console
git clone https://github.com/EgorDudyrev/PeaViner.git
pip install PeaViner/
```

## Running example

### 1. Load and prepare Myocard infarction complication dataset (initially published in UCI repository).
```python3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

df = pd.read_csv('../data/Myocardial_infarction_complications.csv', index_col=0)
df = df.replace('?', np.nan).astype(float)

x_feats = [f for f in df if not f.startswith('Outcome')]
y_feats = df.drop(x_feats, 1).columns
y_feat = df[y_feats[(df[y_feats].nunique()==2)]].mean().idxmax()
print(f"#train features: {len(x_feats)};\t Target feature: '{y_feat}'")

X, y = df[x_feats].values.astype(float), df[y_feat].values.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
print(f"train size: {len(X_train)}; test size: {len(X_test)}")
```
> #train features: 111;	 Target feature: 'Outcome_121. Chronic heart failure (ZSN)' \
> train size: 1360; test size: 340

### 2. Fit and test PeaClassifier model
```python
%%time
from peaviner import PeaClassifier
pc = PeaClassifier()
pc.fit(X_train, y_train, use_tqdm=True)
train_score, test_score = f1_score(y_train, pc.predict(X_train)), f1_score(y_test, pc.predict(X_test))
print(f"Train F1 score: {train_score:.2f}, Test F1 score: {test_score:.2f}")
```
> 100%|██████████| 1241100/1241100 [00:05<00:00, 224558.44it/s] \
> Iter pqr: 100%|██████████| 959/959 [01:30<00:00, 10.65it/s] \
> Iter pq|r: 100%|██████████| 959/959 [02:16<00:00,  7.05it/s] \
> Iter (p|q)r: 100%|██████████| 1574/1574 [01:15<00:00, 20.74it/s] \
> Iter p|q|r: 100%|██████████| 923/923 [00:25<00:00, 36.30it/s]  \
> CPU times: user 5min 42s, sys: 844 ms, total: 5min 43s \
> Wall time: 5min 42s

> Train F1 score: 0.47, Test F1 score: 0.42

### 3. Explain the rule behind the pea classifier 
```python
print(pc.explain([f"'{f}'" for f in x_feats]))
```
> "'2. Age (AGE)' >= 53.0 AND '88. Serum AsAT content (AST_BLOOD) (IU/L)' >= 0.26 OR '12. Presence of chronic Heart failure (HF) in the anamnesis (ZSN_A)' >= 1.0"

This is the only rule a PeaClassifier `pc` uses for prediction. 

### 4. Fit and test CatBoostClassifier for comparison
```python
%%time
from catboost import CatBoostClassifier
cb = CatBoostClassifier()
cb.fit(X_train, y_train, verbose=False)
train_score, test_score = f1_score(y_train, cb.predict(X_train)), f1_score(y_test, cb.predict(X_test))
print(f"Train F1 score: {train_score:.2f}, Test F1 score: {test_score:.2f}")
```
> CPU times: user 6.19 s, sys: 531 ms, total: 6.72 s \
> Wall time: 1.08 s

> Train F1 score: 0.78, Test F1 score: 0.38

### Conclusion
Somehow a PeaClassifier outperforms CatBoost when predicting Chronic heart failure
(test F1 scores are 0.42 vs 0.38 respectively) while using much simpler prediction process.
