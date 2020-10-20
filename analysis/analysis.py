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
df1.to_csv('confronto.csv', index=False)


df2 = {'Data': range(0, len(ydata)), 
       'Ricoverati in ospedale / 8': ydata_ospedale/8, 
      'Terapia intensiva': ydata_terint, 
      'Deceduti x 5': ydata_morti*5}

df2 = pd.DataFrame(df2)
df2.to_csv('serie.csv', index=False)

nomiregioni       = np.array(['Abruzzo','Basilicata','P.A. Bolzano','Calabria','Campania','Emilia-Romagna',
                              'Friuli Venezia Giulia','Lazio','Liguria','Lombardia','Marche','Molise','Piemonte',
                              'Puglia','Sardegna','Sicilia','Toscana','P.A. Trento','Umbria','Valle d\'Aosta','Veneto'])



pop_regioni   =  np.array([1304970, 559084,533050, 1947131, 5801692, 4459477, 1215220, 5879082, 1550640, 10060574, 1525271, 305617, 4356406, 4029053, 1639591, 4999891, 3729641,541380, 882015, 125666, 4905854])


df_popregioni=pd.DataFrame(pop_regioni)
df_popregioni.index=nomiregioni
df_popregioni.columns=['Popolazione']


data_reg2 = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv')

# crea la funzione per confrontare i picchi

def regione_picco(nome_regione):
  mask = data_reg2['denominazione_regione']==nome_regione
  data_reg =data_reg2.loc[mask,:]
  xdatetime=np.array(data_reg['data'])

  xdata_reg=pd.to_numeric(range(data_reg.shape[0]))
  ydata_terint_reg=np.array(data_reg['terapia_intensiva'])
  ydata_ospedale_reg=np.array(data_reg['totale_ospedalizzati'])

  osp_oggi = ydata_ospedale_reg[-1]
  osp_max  = np.max(ydata_ospedale_reg)
  osp_fracpicco = round(osp_oggi/osp_max*100,1)
  osp_datamax = xdatetime[np.argmax(ydata_ospedale_reg)]


  ter_oggi = ydata_terint_reg[-1]
  ter_max  = np.max(ydata_terint_reg)
  ter_fracpicco = round(ter_oggi/ter_max*100,1)
  ter_datamax = xdatetime[np.argmax(ydata_terint_reg)]

  ter_popolazione = round(ter_oggi/df_popregioni.loc[nome_regione,'Popolazione']*10**6,2)

  return [nome_regione,osp_oggi,osp_max,osp_datamax,osp_fracpicco,ter_oggi,ter_max,ter_datamax,ter_fracpicco,ter_popolazione]

# crea il dataframe e lo esporta

# lista regioni da Nord a Sud
lista_regioni = np.array(['Valle d\'Aosta','Liguria','Piemonte','Lombardia','Veneto','Friuli Venezia Giulia','P.A. Bolzano','P.A. Trento',
                          'Emilia-Romagna','Toscana','Marche','Umbria','Abruzzo','Lazio',
                          'Molise','Campania','Puglia','Basilicata','Calabria','Sicilia','Sardegna'])

piccopicco_reg = []

for nomereg in lista_regioni:
  piccopicco_reg.append(regione_picco(nomereg))

piccopicco_reg = pd.DataFrame(piccopicco_reg)

piccopicco_reg.columns=['Nome regione','Ospedalizzati attuali','Ospedalizzati al picco','Data picco osp.','Frazione osp. picco',
                        'Terapia intensiva oggi','Terapia intensiva picco','Data picco terint.','Frazione terint. picco','Terint/popolazione']

piccopicco_reg.to_csv('regioni.csv',index=False)




