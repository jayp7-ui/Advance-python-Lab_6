import pandas as pd
from sklearn.preprocessing import LabelEncoder
#Sample dataframe
data={
    'Education':['High School','Bachelor','Master','Bachelor','Master']
}
df=pd.DataFrame(data)
#Initialize label encoder
le=LabelEncoder()
#Apply Label encoding
df['Education_Encoder']=le.fit_transform(df['Education'])
print(df)