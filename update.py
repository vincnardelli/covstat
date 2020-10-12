import numpy as np
import pandas as pd

# dati cumulativi
 
data = pd.read_csv('https://github.com/pcm-dpc/COVID-19/raw/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv')
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
  
def moving_avg(array,window=7):
    '''This function computes the moving average, given a time window'''
    array_mobile = []
    for i in range(len(array)-window+1):
        mean_parz = np.mean(array[i:i+window])
        array_mobile.append(mean_parz)
        
    return np.array(array_mobile)


def lin_func(t,coeff):
  '''This function takes the coefficient from polyfit'''
  return coeff[1]+t*coeff[0]

ymorti=np.diff(ydata_death)
#ymorti[172]=ymorti[172]-154   # ricalcolo morti Emilia-Romagna
ymorti=np.append([0,0,0,0,0,0,0],ymorti)
 
ydata_morti = moving_avg(ymorti)

kk1=round(ydata_ospedale[-1]/np.max(ydata_ospedale),3)
kk2=round(ydata_terint[-1]/np.max(ydata_terint),3)
kk3=round(ydata_morti[-1]/np.max(ydata_morti),3)

df1 = {'indicatori':['Ricoverati', 'In terapia intensiva', 'Decessi giornalieri'], 
      'Percentuale': [kk1, kk2, kk3], 
      'attuali': [int(ydata_ospedale[-1]), int(ydata_terint[-1]), int(ydata_morti[-1])], 
      'picco': [int(np.max(ydata_ospedale)), int(np.max(ydata_terint)), int(np.max(ydata_morti))]}
df1 = pd.DataFrame(df1)
df1.to_csv('data/confronto.csv', index=False)