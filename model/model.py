import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from scipy.optimize import curve_fit
from scipy.integrate import odeint



# The SIR model differential equations.
def deriv_SIHCDR(y, t, N, beta,gamma,k1,k2,k3,k4,k5):
    S,I,H,C,D,R = y

    dSdt = -(beta*I/N)*S 
    dIdt = (beta*S/N)*I - gamma*I - k1*I
    dHdt = k1*I - (k2+k3)*H
    dCdt = k2*H - (k4+k5)*C
    dDdt = k4*C
    dRdt = k3*H + k5*C + gamma*I 
    
    return dSdt, dIdt, dHdt, dCdt, dDdt, dRdt

def SIHCDR(N,beta,gamma,k1,k2,k3,k4,k5,I0=1,H0=0,C0=0,D0=0,R0=0,t=np.arange(0,365)):
    # Definition of the initial conditions
    # I0 and R0 denotes the number of initial infected people (I0) 
    # and the number of people that recovered and are immunized (R0)
    
    # t ise the timegrid
    
    S0=N-I0-H0-C0-D0-R0  # number of people that can still contract the virus
    
    # Initial conditions vector
    y0 = S0, I0, H0, C0, D0, R0

    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv_SIHCDR, y0, t, args=(N,beta,gamma,k1,k2,k3,k4,k5))
    S, I, H, C, D, R = np.transpose(ret)
    
    return (t,S,I,H,C,D,R)

def moving_avg(array,window=7):
    '''This function computes the moving average, given a time window'''
    array_mobile = []
    for i in range(len(array)-window+1):
        mean_parz = np.mean(array[i:i+window])
        array_mobile.append(mean_parz)
        
    return np.array(array_mobile)



# dati cumulativi
 
data = pd.read_csv('https://github.com/pcm-dpc/COVID-19/raw/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv')
# time threshold
data = data[data['data'] > '2020-10-06']

xdatetime=pd.to_datetime(data['data'])
xdata=np.arange(0,len(data))
ydata=data['totale_casi']
ydata_death=data['deceduti']
ydata_rec=data['dimessi_guariti']
ydata_tamponi=data['tamponi']
ydata_ospedale=np.array(data['totale_ospedalizzati'])
ydata_ricoverati=np.array(data['ricoverati_con_sintomi'])
ydata_terint=np.array(data['terapia_intensiva']) 
ydata_death=np.array(ydata_death)
ydata_deaths = np.diff(ydata_death)
ydata_rec=np.array(ydata_rec)
ydata=np.array(ydata)
 
ydata_inf=np.array(ydata-ydata_rec-ydata_death-ydata_ospedale-ydata_terint)
 
print(data.iloc[-1,:])
 
data.head()

list_death = [ydata_deaths[0]]*6
ydata_deaths_avg = moving_avg(np.append(list_death,ydata_deaths))

def par_scan(vec):

  beta = vec[0]
  gamma = vec[1]
  k1 = vec[2]
  k2 = vec[3]
  k3 = vec[4]
  k4 = vec[5]
  k5 = vec[6]

  fin_res_SIHCRD=SIHCDR(N=60*10**6,
                   beta=beta,
                   gamma=gamma,
                   k1=k1,
                   k2=k2,
                   k3=k3,
                   k4=k4,
                   k5=k5,
                   I0=ydata_inf[0],D0=ydata_death[0],R0=7*10**6,H0=ydata_ospedale[0],C0=ydata_terint[0])

  time=fin_res_SIHCRD[0]

  # model predictions
  ymodel_inf = fin_res_SIHCRD[2][0:len(ydata_inf)]
  ymodel_ospedale = fin_res_SIHCRD[3][0:len(ydata_inf)]
  ymodel_terint = fin_res_SIHCRD[4][0:len(ydata_inf)]
  ymodel_deaths = np.diff(fin_res_SIHCRD[5][0:len(ydata_inf)])

  # comparison with true data
  delta_inf = round(np.mean(np.abs(ymodel_inf - ydata_inf)/ydata_inf)*100,2)
  delta_ospedale = round(np.mean(np.abs(ymodel_ospedale - ydata_ospedale)/ydata_ospedale)*100,2)
  delta_terint = round(np.mean(np.abs(ymodel_terint - ydata_terint)/ydata_terint)*100,2)
  delta_deaths = round(np.mean(np.abs(ymodel_deaths - ydata_deaths)/ydata_deaths)*100,2)

  delta_avg = round((delta_inf+delta_ospedale+delta_terint+delta_deaths)/4,2)

  print(delta_inf,delta_ospedale,delta_terint,delta_deaths,delta_avg)

  return delta_avg

#x0=np.array([0.5,0.3,1/5, 1/10, 1/10, 1/30, 1/20])
x0=np.array([0.5,0.3,0.1,0.1,0.1,0.1,0.1])

par_scan(x0)



from scipy.optimize import minimize

res = minimize(par_scan,x0,method='nelder-mead',options={'max_iter':1000,'disp': True,'adaptive':True})
 
print(res.message)
print(res.x)





x_opt = res.x
 
fin_res_SIHCRD=SIHCDR(N=60*10**6,
                   beta=x_opt[0],
                   gamma=x_opt[1],
                   k1=x_opt[2],
                   k2=x_opt[3],
                   k3=x_opt[4],
                   k4=x_opt[5],
                   k5=x_opt[6],
                   I0=ydata_inf[0],D0=ydata_death[0],R0=7*10**6,H0=ydata_ospedale[0],C0=ydata_terint[0])
 
time=fin_res_SIHCRD[0]
 
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(xdata,ydata_inf/1000,'.',label='Attualmente positivi /1000',color='orange')
plt.plot(xdata,ydata_ospedale/100,'.',label='Ospedalizzati /100',color='green')
plt.plot(xdata,ydata_terint/10,'.', label='In terapia intensiva /10',color='red')
plt.plot(xdata[1:],ydata_deaths,'.',label='Decessi',color='black')
plt.plot(time,fin_res_SIHCRD[2]/1000,color='orange',linestyle='--')
plt.plot(time,fin_res_SIHCRD[3]/100,color='green',linestyle='--')
plt.plot(time,fin_res_SIHCRD[4]/10,color='red',linestyle='--')
plt.plot(time[1:],np.diff(fin_res_SIHCRD[5]),color='black',linestyle='--')
plt.ylim(0,200)
#plt.xticks(np.arange(0,360,30),['Ott','Nov','Dic','Gen','Feb','Mar','Apr','Mag','Giu','Lug','Ago','Set'])
plt.xlim(0,30)
plt.xlabel('Giorni dal 6 Ottobre')
plt.grid()
plt.legend()
plt.subplot(1,2,2)
plt.plot(xdata,ydata_inf/1000,'.',label='Attualmente positivi /1000',color='orange')
plt.plot(xdata,ydata_ospedale/100,'.',label='Ospedalizzati /100',color='green')
plt.plot(xdata,ydata_terint/10,'.', label='In terapia intensiva /10',color='red')
plt.plot(xdata[1:],ydata_deaths,'.',label='Decessi',color='black')
plt.text(50,670,'Limite posti in terapia intensiva',color='red')
plt.plot(time,fin_res_SIHCRD[4]*0+650,color='red')
plt.plot(time,fin_res_SIHCRD[2]/1000,color='orange',linestyle='--')
plt.plot(time,fin_res_SIHCRD[3]/100,color='green',linestyle='--')
plt.plot(time,fin_res_SIHCRD[4]/10,color='red',linestyle='--')
plt.plot(time[1:],np.diff(fin_res_SIHCRD[5]),color='black',linestyle='--')
plt.plot(time,fin_res_SIHCRD[1]/100000,color='blue',label='Suscettibili / 100000',linestyle='--')
plt.ylim(10,800)
plt.xticks(np.arange(0,360,30),['Ott','Nov','Dic','Gen','Feb','Mar','Apr','Mag','Giu','Lug','Ago','Set'])
plt.xlim(0,210)
plt.grid()
plt.legend()
plt.show()



x_opt

print('Frazione della popolazione che ha contratto il virus a fine epidemia ',round((60*10**6-fin_res_SIHCRD[1][-1])/(60*10**6)*100),'%')

print('Numero di persone contagiate a fine epidemia ',(60*10**6-fin_res_SIHCRD[1][-1])*10**-6,'milioni')

print('Numero decessi seconda ondata ',round(fin_res_SIHCRD[5][1]-ydata_deaths[0]))

# export section for github action
decessi = np.diff(fin_res_SIHCRD[5])
decessi = np.insert(decessi, 0, 28, axis=0)


df = {'time': time, 
      'terapia_intensiva': fin_res_SIHCRD[4], 
      'ospedalizzati/10': fin_res_SIHCRD[3]/10, 
      'attualmente_positivi/100': fin_res_SIHCRD[2]/100, 
      'decessi': decessi, 
      'limite_ti': fin_res_SIHCRD[4]*0+6500}
df = pd.DataFrame(df)
df.drop(df.index[(len(ydata_inf)+30):], inplace=True)
start = datetime.datetime.strptime("6-10-2020", "%d-%m-%Y")


df['date'] = pd.date_range(start, periods=len(ydata_inf)+30)

df.to_csv('modello.csv', index=False)
df


