import pandas as pd
import numpy as np
data={
    'Value':[1,10,100,1000]
}
df=pd.DataFrame(data)
#Apply log10 transformation
df['Log_Value']=np.log10(df['Value'])
print(df)

#Apply square root transformation
df['Sqrt_Value']=np.sqrt(df['Value'])
print(df)

from sklearn.preprocessing import StandardScaler

data=pd.DataFrame({'Value':[50,100,150]})
scaler= StandardScaler()
data['Standardized']=scaler.fit_transform(data[['Value']])
print(data)