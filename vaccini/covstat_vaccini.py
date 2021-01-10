import pandas as pd
import numpy as np

vaccinations = pd.read_csv("https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv")

stati = ["Germany", "Italy"]
df = vaccinations[vaccinations.location.isin(stati)]
df = df.pivot(index='date', columns='location', values=['total_vaccinations', 'total_vaccinations_per_hundred'])
df.columns = df.columns.map('_'.join).str.strip('')
df.reset_index(inplace=True)
df.date = pd.to_datetime(df.date).dt.strftime('%d-%m-%Y')
df['spread'] = df['total_vaccinations_per_hundred_Germany'] - df['total_vaccinations_per_hundred_Italy']
df.dropna(axis=0, inplace = True)
df['total_vaccinations_Germany'] = df['total_vaccinations_Germany'].astype(int)
df['total_vaccinations_Italy'] = df['total_vaccinations_Italy'].astype(int)
df['spread'] = round(df['spread'], 2)
df.tail(1).to_csv("confronto_italia_germania_last.csv")

df = vaccinations[vaccinations.location.isin(stati)]
df.drop(columns=['iso_code', 'total_vaccinations', 'daily_vaccinations', 'daily_vaccinations_per_million'], inplace=True)
df = df.melt(id_vars = ['location', 'date'])
df.reset_index(inplace=True)
df.date = pd.to_datetime(df.date).dt.strftime('%d-%m-%Y')
df.to_csv("confronto_italia_germania.csv")
df

stati = ['Italy', 'Germany', 'United Kingdom', 'Spain', 'United States']
df = vaccinations[vaccinations.location.isin(stati)]
df.drop(columns=['iso_code'], inplace=True)
df = df.melt(id_vars = ['location', 'date'])
df = df.pivot(index = ['date', 'variable'], columns = 'location', values= 'value')
df.reset_index(inplace=True)
df.date = pd.to_datetime(df.date).dt.strftime('%d-%m-%Y')
df.loc[df['variable'] == 'daily_vaccinations', ['variable']] = "Vaccinazioni giornaliere"
df.loc[df['variable'] == 'daily_vaccinations_per_million', ['variable']] = "Vaccinazioni giornaliere ogni milione ab"
df.loc[df['variable'] == 'total_vaccinations', ['variable']] = "Totale vaccinazioni"
df.loc[df['variable'] == 'total_vaccinations_per_hundred', ['variable']] = "Totale vaccinazioni ogni cento abitanti"

df.to_csv("confronto_stati.csv")
df


anagrafica = pd.read_csv("https://raw.githubusercontent.com/italia/covid19-opendata-vaccini/master/dati/anagrafica-vaccini-summary-latest.csv")
dati_istat = pd.read_csv("https://raw.githubusercontent.com/vincnardelli/covstat/master/vaccini/dati_istat.csv", sep=";")
dati_istat

anagrafica = anagrafica.merge(dati_istat, on="fascia_anagrafica")
anagrafica['Maschi'] = anagrafica['sesso_maschile']/anagrafica['pop_maschile']
anagrafica['Femmine'] = anagrafica['sesso_femminile']/anagrafica['pop_femminile']

anagrafica_graph = anagrafica[['fascia_anagrafica','Maschi', 'Femmine']]
anagrafica_graph.set_index('fascia_anagrafica', inplace = True)
anagrafica_graph['Femmine'] = anagrafica_graph['Femmine']*-100
anagrafica_graph['Maschi'] = anagrafica_graph['Maschi']*100
anagrafica_graph = round(anagrafica_graph, 2)
anagrafica_graph.to_csv("italia_anagrafica.csv")
anagrafica_graph

popolation_regions = np.array([ 1304970,      559084,        533050,   1947131,   5801692,         
                               4459477,        1215220,5879082, 1550640,    
                               10060574,  1525271,  305617,    4356406, 4029053, 1639591,  
                               4999891,  3729641,       541380,  882015,          125666, 4905854])
name_regions       = np.array(['Abruzzo','Basilicata','P.A. Bolzano','Calabria','Campania',
                               'Emilia-Romagna','Friuli Venezia Giulia','Lazio','Liguria',
                               'Lombardia','Marche','Molise','Piemonte','Puglia','Sardegna',
                               'Sicilia','Toscana','P.A. Trento','Umbria','Valle d\'Aosta','Veneto'])
area       = np.array(['ABR','BAS','PAB','CAL','CAM',
                               'EMR','FVG','LAZ','LIG',
                               'LOM','MAR','MOL','PIE','PUG','SAR',
                               'SIC','TOS','PAT','UMB','VDA','VEN'])

popolation = pd.DataFrame([name_regions, popolation_regions, area]).transpose()
popolation.columns = ['regione', 'popolazione', 'area'] 

regioni = pd.read_csv("https://raw.githubusercontent.com/italia/covid19-opendata-vaccini/master/dati/somministrazioni-vaccini-summary-latest.csv")
regioni = regioni.merge(popolation, on='area')

regioni['Totale vaccinazioni'] = regioni.groupby(['regione'])['totale'].apply(lambda x: x.cumsum())
regioni['Totale vaccinazioni ogni cento ab'] = regioni['Totale vaccinazioni']/regioni['popolazione']*100
regioni['Totale vaccinazioni ogni cento ab'] = round(regioni['Totale vaccinazioni ogni cento ab'].astype(float), 2)


regioni['Vaccinazioni giornaliere ogni cento ab'] = regioni['totale']/regioni['popolazione']*100
regioni['Vaccinazioni giornaliere ogni cento ab'] = round(regioni['Vaccinazioni giornaliere ogni cento ab'].astype(float), 2)

regioni = regioni.rename(columns={'data_somministrazione': 'data', 
                        'totale': 'Vaccinazioni giornaliere', 
                        'categoria_operatori_sanitari_sociosanitari': 'Operatori sanitari - sociosanitari', 
                        'categoria_personale_non_sanitario': 'Personale non sanitario', 
                        'categoria_ospiti_rsa': 'Ospiti RSA', 
                        'regione': 'Regione'})
#regioni.reset_index(inplace=True)
#regioni.data = pd.to_datetime(regioni.data).dt.strftime('%d-%m-%Y')
regioni.to_csv("regioni.csv")

confronto_regioni = regioni[['data', 'Regione', 'Totale vaccinazioni', 'Totale vaccinazioni ogni cento ab', 'Vaccinazioni giornaliere', 'Vaccinazioni giornaliere ogni cento ab']]
confronto_regioni = confronto_regioni.melt(id_vars = ['Regione', 'data'])
confronto_regioni = confronto_regioni.pivot(index = ['data', 'variable'], columns = 'Regione', values= 'value')
confronto_regioni.reset_index(inplace=True)
confronto_regioni.data = pd.to_datetime(confronto_regioni.data).dt.strftime('%m-%d-%Y')
confronto_regioni.to_csv("confronto_regioni.csv")

confronto_regioni
