import pandas as pd
data = {'name': ['test'], 'number': ['3']}
df = pd.DataFrame(data=data)
df.to_csv("df.csv")
print(df)