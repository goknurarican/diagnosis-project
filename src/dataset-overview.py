import pandas as pd
import matplotlib.pyplot as plt

df_train = pd.read_csv('../data/ddxplus/train.csv', engine='python', quotechar='"')
df_validate = pd.read_csv('../data/ddxplus/validate.csv', engine='python', quotechar='"')
df_test = pd.read_csv('../data/ddxplus/test.csv', engine='python', quotechar='"')

counts = {'train': len(df_train), 'validate': len(df_validate), 'test': len(df_test)}
plt.figure()
plt.bar(counts.keys(), counts.values())
plt.title('number of records per split')
plt.ylabel('amount:')
plt.show()

#missing values (percentage) in train set
missing = df_train.isna().mean() * 100
plt.figure()
plt.bar(missing.index, missing.values)
plt.title('missing balue percentage (train)')
plt.xticks(rotation=45, ha='right')
plt.ylabel('%')
plt.show()

#top 10 pathologies in train set
top10 = df_train['PATHOLOGY'].value_counts().head(10)
plt.figure()
plt.bar(top10.index, top10.values)
plt.title('Top 10 pathologies (train)')
plt.xticks(rotation=45, ha='right')
plt.ylabel('amount')
plt.show()

plt.figure()
plt.hist(df_train['AGE'].dropna(), bins=20)
plt.title('Age distribution (train)')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

sex_counts = df_train['SEX'].value_counts()
plt.figure()
plt.bar(sex_counts.index, sex_counts.values)
plt.title('Sex distribution (train)')
plt.ylabel('amoınt')
plt.show()
