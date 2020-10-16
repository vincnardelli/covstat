import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#from google.colab import drive
#drive.mount('/content/gdrive')

#cd gdrive/My Drive/Colab Notebooks/results

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
 
print(data.iloc[-1,:])
 
data.head()

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

print('ospedalizzati attuali',ydata_ospedale[-1],'ospedalizzati massimo',np.max(ydata_ospedale))
print('terapia intensiva attuali',ydata_terint[-1],'terapia intensiva massimo',np.max(ydata_terint))
print('decessi giornalieri attuali',round(ydata_morti[-1]),'decessi giornalieri massimo',round(np.max(ydata_morti)))

print(kk1,kk2,kk3)
 
plt.barh([0,1,2],[kk3,kk2,kk1])
plt.xlim(0,1)
plt.yticks([2,1,0],['Ricoverati','In terapia intensiva','Decessi giornalieri'],rotation=0)
plt.xticks(np.arange(0.1,1.1,0.1),['10%','20%','30%','40%','50%','60%','70%','80%','90%','100%'])
plt.title('Confronto con il picco della prima ondata')
plt.grid()
plt.show()

# confronto curve ricoverati, terapia intensiva e deceduti
 
plt.plot(ydata_ospedale/8,color='green',label='Ricoverati in ospedale /8')
plt.plot(ydata_terint,color='red',label='Ricoverati in terapia intensiva')
plt.plot(ydata_morti*5,color='black',label='Deceduti x5')
plt.xticks(np.arange(0,250,7),['24 Feb','2 Mar','9 Mar','16 Mar','23 Mar','30 Mar','6 Apr','13 Apr','20 Apr','27 Apr','4 Mag','11 Mag','18 Mag','25 Mag','1 Giu','8 Giu','15 Giu','22 Giu','29 Giu','6 Lug','13 Lug','20 Lug','27 Lug','3 Ago','10 Ago','17 Ago','24 Ago','31 Ago','7 Set','14 Set','21 Set','28 Set','5 Ott','12 Ott','19 Ott','26 Ott'],rotation=70)
#plt.ylim(0,200)
plt.xlim(0,len(ydata_terint))
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('terint_plot.png',dpi=300)
plt.show()

# last period analysis

y1 = ydata_ospedale[-30:]/8
y2 = ydata_terint[-30:]
y3 = ydata_morti[-30:]*5

c1 = np.polyfit(np.arange(0,30),y1,1)
c2 = np.polyfit(np.arange(0,30),y2,1)
c3 = np.polyfit(np.arange(0,30),y3,1)

tgrid = np.arange(0,30)
y1fit=lin_func(tgrid,c1)
y2fit=lin_func(tgrid,c2)
y3fit=lin_func(tgrid,c3)

plt.plot(y1,'.g',label='Ricoverati in ospedale /8 (dati)')
plt.plot(y2,'.r',label='Ricoverati in terapia intensiva (dati)')
plt.plot(y3,'.k',label='Deceduti x5 (dati)')
plt.plot(y1fit,color='green',label='Ricoverati in ospedale /8 (fit)',linestyle='--')
plt.plot(y2fit,color='red',label='Ricoverati in terapia intensiva (fit)',linestyle='--')
plt.plot(y3fit,color='black',label='Deceduti x5 (fit)',linestyle='--')
#plt.plot(ydata_morti*5,color='black',label='Deceduti x5')
#plt.xticks(np.arange(0,250,7),['24 Feb','2 Mar','9 Mar','16 Mar','23 Mar','30 Mar','6 Apr','13 Apr','20 Apr','27 Apr','4 Mag','11 Mag','18 Mag','25 Mag','1 Giu','8 Giu','15 Giu','22 Giu','29 Giu','6 Lug','13 Lug','20 Lug','27 Lug','3 Ago','10 Ago','17 Ago','24 Ago','31 Ago','7 Set','14 Set','21 Set','28 Set','5 Ott'],rotation=70)
plt.ylim(0,700)
plt.xlabel('Last 30 days')
#plt.xlim(len(ydata_terint)-30,len(ydata_terint))
plt.grid()
plt.legend()
plt.tight_layout()
#plt.savefig('output3/terint_plot.png',dpi=300)
plt.show()



# attualmente positivi

plt.plot(ydata_inf,color='red',label='Attualmente positivi')
plt.xticks(np.arange(0,250,7),['24 Feb','2 Mar','9 Mar','16 Mar','23 Mar','30 Mar','6 Apr','13 Apr','20 Apr','27 Apr','4 Mag','11 Mag','18 Mag','25 Mag','1 Giu','8 Giu','15 Giu','22 Giu','29 Giu','6 Lug','13 Lug','20 Lug','27 Lug','3 Ago','10 Ago','17 Ago','24 Ago','31 Ago','7 Set','14 Set','21 Set','28 Set','5 Ott'],rotation=70)
#plt.ylim(0,200)
plt.xlim(0,len(ydata_inf))
plt.grid()
plt.legend()
plt.tight_layout()
#plt.savefig('output3/terint_plot.png',dpi=300)
plt.show()

yt=np.diff(ydata_tamponi[-120:])
yp=np.diff(ydata[-120:])
 
print('Coefficiente di correlazione ',np.corrcoef(yt**2,yp)[0,1])
 
# fit quadratico
coef = np.polyfit(yt,yp,2)
x=1000*np.arange(0,150,1)
yfit= coef[2]+coef[1]*x+coef[0]*x**2
 
plt.plot(yt,yp,'.',label='dati')
plt.plot(x,yfit,label='fit quadratico')
plt.xlabel('Numero tamponi')
plt.ylabel('Numero positivi')
plt.grid()
plt.ylim(0,)
plt.xlim(20000,140000)
plt.title('Dati degli ultimi 4 mesi. Correlazione = 88%')
plt.legend(loc=2)
plt.savefig('corr_tamponi_infetti.png',dpi=300)
plt.show()

frac_tamponi_positivi = np.append([0,0,0,0,0,0,0],np.diff(ydata)/np.diff(ydata_tamponi))
frac_tamponi_positivi_avg = moving_avg(frac_tamponi_positivi)*100

plt.plot(frac_tamponi_positivi_avg)
plt.grid()
plt.yticks(np.arange(0,31,5),['0%','5%','10%','15%','20%','25%','30%'])
plt.xticks(np.arange(0,250,14),['24 Feb','9 Mar','23 Mar','6 Apr','20 Apr','4 Mag','18 Mag','1 Giu','15 Giu','29 Giu','13 Lug','27 Lug','10 Ago','24 Ago','7 Set','21 Set','5 Ott','19 Ott'],rotation=70)
plt.title('Frazione di tamponi positivi')
plt.show()

# sommario

plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
plt.barh([0,1,2],[kk3,kk2,kk1],color='green')
plt.xlim(0,1)
plt.yticks([2,1,0],['Ricoverati','In terapia intensiva','Decessi giornalieri \n (media mobile 7 giorni)'],rotation=0)
plt.xticks(np.arange(0.1,1.1,0.1),['10%','20%','30%','40%','50%','60%','70%','80%','90%','100%'])
plt.title('Confronto con il picco della prima ondata')
plt.grid()
plt.subplot(2,2,2)
plt.plot(ydata_ospedale/8,color='green',label='Ricoverati in ospedale /8')
plt.plot(ydata_terint,color='red',label='Ricoverati in terapia intensiva')
plt.plot(ydata_morti*5,color='black',label='Deceduti x5 (media mobile 7 giorni)')
plt.xticks(np.arange(0,250,14),['24 Feb','9 Mar','23 Mar','6 Apr','20 Apr','4 Mag','18 Mag','1 Giu','15 Giu','29 Giu','13 Lug','27 Lug','10 Ago','24 Ago','7 Set','21 Set','5 Ott','19 Ott'],rotation=70)
plt.xlim(0,len(ydata_terint))
plt.grid()
plt.legend()
plt.subplot(2,2,3)
plt.plot(y1,'g',marker='.',label='Ricoverati in ospedale /8')
plt.plot(y2,'r',marker='.',label='Ricoverati in terapia intensiva')
plt.plot(y3,'k',marker='.',label='Deceduti x5 (media mobile 7 giorni)')
#plt.plot(y1fit,color='green',label='Ricoverati in ospedale /10 (fit)',linestyle='--')
#plt.plot(y2fit,color='red',label='Ricoverati in terapia intensiva (fit)',linestyle='--')
#plt.plot(y3fit,color='black',label='Deceduti x5 (fit)',linestyle='--')
plt.annotate('Segnale di allarme',xy=(22,450),xytext=(5,480),arrowprops={'arrowstyle':'->'},fontsize=14,color='purple')
plt.xticks(np.arange(0,30),np.arange(30,0,-1),rotation=90)
plt.xlim(-0.5,29.5)
plt.ylim(0,650)
plt.xlabel('Ultimi 30 giorni')
plt.grid()
plt.legend()
plt.subplot(2,2,4)
plt.plot(frac_tamponi_positivi_avg,color='brown')
plt.grid()
plt.yticks(np.arange(0,31,5),['0%','5%','10%','15%','20%','25%','30%'])
plt.xticks(np.arange(0,250,14),['24 Feb','9 Mar','23 Mar','6 Apr','20 Apr','4 Mag','18 Mag','1 Giu','15 Giu','29 Giu','13 Lug','27 Lug','10 Ago','24 Ago','7 Set','21 Set','5 Ott','19 Ott'],rotation=70)
plt.title('Frazione di tamponi positivi (media mobile 7 giorni)')
plt.xlim(0,len(frac_tamponi_positivi_avg))
plt.tight_layout()
plt.savefig('multi_comparison.png',dpi=300)
plt.show()

nomiregioni       = np.array(['Abruzzo','Basilicata','P.A. Bolzano','Calabria','Campania','Emilia-Romagna',
                              'Friuli Venezia Giulia','Lazio','Liguria','Lombardia','Marche','Molise','Piemonte',
                              'Puglia','Sardegna','Sicilia','Toscana','P.A. Trento','Umbria','Valle d\'Aosta','Veneto'])



pop_regioni   =  np.array([1304970, 559084,533050, 1947131, 5801692, 4459477, 1215220, 5879082, 1550640, 10060574, 1525271, 305617, 4356406, 4029053, 1639591, 4999891, 3729641,541380, 882015, 125666, 4905854])


df_popregioni=pd.DataFrame(pop_regioni)
df_popregioni.index=nomiregioni
df_popregioni.columns=['Popolazione']

df_popregioni

#nome_regione=input('Scegli la regione: ')
nome_regione='Abruzzo'

popolazione_regione=df_popregioni.loc[str(nome_regione),'Popolazione']

print(popolazione_regione)

# dati cumulativi

data_reg = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv')
mask = data_reg['denominazione_regione']==nome_regione
data_reg =data_reg.loc[mask,:]

xdata_reg=pd.to_numeric(range(data_reg.shape[0]))
ydata_reg=np.array(data_reg['totale_casi'])
ydata_death_reg=np.array(data_reg['deceduti'])
ydata_rec_reg=np.array(data_reg['dimessi_guariti'])

ydata_inf_reg=np.array(ydata_reg-ydata_rec_reg-ydata_death_reg)

# i dati iniziano all'ottavo giorno. Prima non ci sono contagi
delay=0
ydata_reg=ydata_reg[delay:]
ydata_death_reg=ydata_death_reg[delay:]
ydata_rec_reg=ydata_rec_reg[delay:]
ydata_inf_reg=ydata_inf_reg[delay:]

print(data_reg.iloc[-1,:])

data_reg.head()

ydata_terint_reg=np.array(data_reg['terapia_intensiva'])
ydata_ospedale_reg=np.array(data_reg['totale_ospedalizzati'])
ydata_tamponi_reg=np.array(data_reg['tamponi'])

positivi_giorn_reg=np.diff(ydata_reg)
tamponi_giorn_reg=np.diff(ydata_tamponi_reg)

ymorti_reg_avg = moving_avg(np.append([0,0,0,0,0,0,0],np.diff(ydata_death_reg)))

plt.plot(np.arange(len(ydata_reg)),ydata_ospedale_reg,label='Ospedalizzati',color='green')
plt.plot(np.arange(len(ydata_reg)),ydata_terint_reg,label='In terapia intensiva',color='red')
plt.plot(np.arange(len(ydata_reg)),ymorti_reg_avg,label='Deceduti (media mobile 7 giorni)',color='black')
plt.xticks(np.arange(0,250,14),['24 Feb','9 Mar','23 Mar','6 Apr','20 Apr','4 Mag','18 Mag','1 Giu','15 Giu','29 Giu','13 Lug','27 Lug','10 Ago','24 Ago','7 Set','21 Set','5 Ott','19 Ott'],rotation=70)
plt.xlim(0,len(ydata_reg))
plt.ylabel('Totale')
plt.title('Regione '+str(nome_regione))
plt.legend()
plt.grid()
#plt.savefig('output3_regioni/dataonly.png')
plt.show()

file_confirmed='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
file_deaths='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
file_recovered='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'

df_confirmed=pd.read_csv(file_confirmed)
df_deaths=pd.read_csv(file_deaths)
df_recovered=pd.read_csv(file_recovered)

df_deaths.head()

def daily_infected(country):
    df_confirmed_country = df_confirmed[df_confirmed['Country/Region']==country].iloc[:,4:]
    ydata_inf =  np.sum(np.diff(np.array(df_confirmed_country)),axis=0)
    xdata_inf =  pd.to_datetime(df_confirmed_country.columns[1:],infer_datetime_format=True)
    return xdata_inf,ydata_inf

def daily_deaths(country):
    df_deaths_country = df_deaths[df_deaths['Country/Region']==country].iloc[:,4:]
    ydata_inf =  np.sum(np.diff(np.array(df_deaths_country)),axis=0)
    xdata_inf =  pd.to_datetime(df_deaths_country.columns[1:],infer_datetime_format=True)
    return xdata_inf,ydata_inf

def daily_recovered(country):
    df_recovered_country = df_recovered[df_recovered['Country/Region']==country].iloc[:,4:]
    ydata_inf =  np.sum(np.diff(np.array(df_recovered_country)),axis=0)
    xdata_inf =  pd.to_datetime(df_recovered_country.columns[1:],infer_datetime_format=True)
    return xdata_inf,ydata_inf

def daily_deaths_avg(country,window_mean=7):
    df_deaths_country = df_deaths[df_deaths['Country/Region']==country].iloc[:,4:]
    ydata_deaths = np.sum(np.diff(np.array(df_deaths_country)),axis=0)
    xdata_deaths = pd.to_datetime(df_deaths_country.columns[1:],infer_datetime_format=True)
    ydata_deaths_mean = moving_avg(ydata_deaths,window=window_mean)
    return xdata_deaths[window_mean-1:],ydata_deaths_mean,xdata_deaths,ydata_deaths

#  single country
select_country = 'Italy'    # you can choose the country here

plt.plot(daily_deaths_avg(select_country)[0],daily_deaths_avg(select_country)[1],label=select_country)
plt.xticks(rotation=70)
plt.ylim(0,)
plt.grid()
plt.ylabel('Decessi giornalieri')
plt.title('Media mobile 7 giorni')
plt.legend()
plt.show()

# list of countries
list_of_countries = ['Italy','Germany','Spain','France','US','United Kingdom']

for country in list_of_countries:
    plt.plot(daily_deaths_avg(country)[0],daily_deaths_avg(country)[1],label=country)
plt.xticks(rotation=70)
plt.ylim(0,)
plt.grid()
plt.ylabel('Decessi giornalieri')
plt.title('Media mobile 7 giorni')
plt.legend()
plt.show()

# with a list of countries, normalized to the population

list_of_countries2 = ['Italy','Germany','Spain','France','US','United Kingdom','Sweden']
population_dict = {'Italy':60.36*10**6,
                  'Germany':83.02*10**6,
                  'Spain':46.94*10**6,
                  'France':66.99*10**6,
                  'US':328.2*10**6,
                  'United Kingdom':66.65*10**6,
                  'Sweden':10.23*10**6}

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(daily_deaths_avg('Italy')[0][30:],daily_deaths_avg('Italy')[1][30:],label='Italy')
plt.plot(daily_deaths_avg('Germany')[0][30:],daily_deaths_avg('Germany')[1][30:],label='Germany')
plt.xticks(rotation=70)
plt.ylim(0,)
plt.grid()
plt.ylabel('Decessi giornalieri')
plt.title('Media mobile 7 giorni')
plt.legend()
plt.subplot(1,2,2)
for country in list_of_countries2:
    plt.plot(daily_deaths_avg(country)[0][30:],daily_deaths_avg(country)[1][30:]/population_dict[country]*10**6,label=country)
plt.xticks(rotation=70)
plt.ylim(0,)
plt.grid()
plt.ylabel('Decessi giornalieri per milione di abitanti')
plt.title('Media mobile 7 giorni')
plt.legend()
plt.tight_layout()
plt.savefig('deaths_report.png',dpi=300)
plt.show()



plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(daily_infected('Italy')[0],daily_infected('Italy')[1],label='Italy',color='blue')
plt.legend()
plt.ylim(0,8000)
plt.grid()
plt.xticks(rotation=60)
plt.subplot(1,2,2)
plt.plot(daily_infected('Germany')[0],daily_infected('Germany')[1],label='Germany',color='black')
plt.legend()
plt.grid()
plt.ylim(0,8000)
plt.xticks(rotation=60)
plt.show()

plt.plot(daily_infected('Italy')[0],np.cumsum(daily_infected('Italy')[1]),label='Infetti')
plt.plot(daily_infected('Italy')[0],np.cumsum(daily_recovered('Italy')[1]),label='Guariti')
plt.plot(daily_infected('Italy')[0],np.cumsum(daily_deaths('Italy')[1]),label='Deceduti')
plt.legend()
plt.xticks(rotation=70)
plt.ylabel('Totale')
plt.title('Italia')
plt.grid()
plt.show()



from scipy.integrate import odeint



# The SIR model differential equations.
def deriv_SIR(y, t, N, beta,gamma):
    S,I,R = y

    dSdt = -(beta*I/N)*S 
    dIdt = (beta*S/N)*I - gamma*I 
    dRdt = gamma*I 
    
    return dSdt, dIdt, dRdt

def SIR(N,beta,gamma,I0=1,R0=0,t=np.arange(0,365)):
    # Definition of the initial conditions
    # I0 and R0 denotes the number of initial infected people (I0) 
    # and the number of people that recovered and are immunized (R0)
    
    # t ise the timegrid
    
    S0=N-I0-R0  # number of people that can still contract the virus
    
    # Initial conditions vector
    y0 = S0, I0, R0

    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv_SIR, y0, t, args=(N,beta,gamma))
    S, I, R = np.transpose(ret)
    
    return (t,S,I,R)



r0_example=4

fin_res_SIR1=SIR(16000,0.1*r0_example,0.1,I0=1,R0=0)
t_vec=fin_res_SIR1[0]
s_vec=fin_res_SIR1[1]
i_vec=fin_res_SIR1[2]
r_vec=fin_res_SIR1[3]

plt.plot(t_vec,s_vec,color='royalblue',label='Susceptible')
plt.plot(t_vec,i_vec,color='red',label='Infected')
plt.plot(t_vec,r_vec,color='green',label='Removed')
#plt.plot(t_vec,r0_example*s_vec/1000*100,color='black',label='R0 x 100',linestyle='--')
plt.xlim(0,200)
#plt.ylim(0,500)
plt.xlabel('Days from the beginning')
plt.ylabel('Total number')
plt.title('R0 = '+str(r0_example))
plt.legend()
plt.grid()
plt.tight_layout()
#plt.savefig('SIR_example.png',dpi=300)
plt.show()

# The SIR model differential equations.
def deriv_SIR_2(y, t, N, beta1,gamma,tau=10**6,t_thresh=14):
    S,I,R = y
    
    if t<=t_thresh:      # il lockdown nazionale inizia al 14° giorno 
        B=beta1
    else: 
        B=beta1*np.exp(-(t-t_thresh)/tau)

    dSdt = -(B*I/N)*S 
    dIdt = (B*S/N)*I - gamma*I 
    dRdt = gamma*I 
    
    return dSdt, dIdt, dRdt


def SIR_2(N,beta1,gamma,tau=10**6,t_thresh=14,I0=1,R0=0,t=np.arange(0,365)):
    # Definition of the initial conditions
    # I0 and R0 denotes the number of initial infected people (I0) 
    # and the number of people that recovered and are immunized (R0)
    
    # t ise the timegrid
    
    S0=N-I0-R0  # number of people that can still contract the virus
    
    # Initial conditions vector
    y0 = S0, I0, R0

    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv_SIR_2, y0, t, args=(N,beta1,gamma,tau,t_thresh))
    S, I, R = np.transpose(ret)
    
    return (t,S,I,R)



# prova aggiunta codice 

df_deaths.head()


