import numpy as np
import pandas as pd

data = pd.read_csv('https://github.com/pcm-dpc/COVID-19/raw/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv')

print("data loaded")
xdata=pd.to_numeric(range(data.shape[0]))
ydata=data['totale_casi']
ydata_death=data['deceduti']
ydata_rec=data['dimessi_guariti']
ydata_tamponi=data['tamponi']
ydata_ospedale=np.array(data['totale_ospedalizzati'])
ydata_ricoverati=np.array(data['ricoverati_con_sintomi'])
 
ydata_death=np.array(ydata_death)
ydata_rec=np.array(ydata_rec)
ydata=np.array(ydata)
 
ydata_inf=np.array(ydata-ydata_rec-ydata_death)
 
ydata_terint=np.array(data['terapia_intensiva'])
  
newdata = {'x': xdata, 
'totale_casi':ydata, 
'deceduti': ydata_death, 
'rec': ydata_rec}

newdata = pd.DataFrame(newdata)
print(newdata.head())

newdata.to_csv("df.csv")
print("csv export")
