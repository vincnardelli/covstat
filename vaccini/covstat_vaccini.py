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


