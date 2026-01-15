#!/usr/bin/env python
# coding: utf-8

# # Detección de _Runaway Electrons_ en GOLEM

# Los _runaway electrons_ son electrones acelerados a velocidades relativistas. Estos se han observado en plasmas de alta energía. En los tokamaks, la aceleración de estos electrones ocurre cuando la fuerza del campo eléctrico supera a la fuerza de frenado producida por colisiones con otras partículas.

# El valor crítico del campo eléctrico bajo el cual no se producen _runaway electrons_ es el siguiente,

# $$
# E_c = \frac{e^3n_eln \Lambda}{4\pi \varepsilon_o^2 m_e c^2}
# $$

# donde $n_e$ es la densidad electrónica y $ln\Lambda$ es el logaritmo de Coulomb [1].

# Este tipo de electrones inducen calor de forma localizada en las paredes del tokamak, lo cual puede llegar a dañar las componentes más frágiles del dispositivo. Es por eso que es escencial prevenir que se generen o mitigarlos si se llegan a producir.

# El tokamak GOLEM cuenta con un mecanismo de detección de _runaway electrons_ mediante el diagnóstico _Scintillation probes_ (Sondas de centelleo). Este diagnóstico cuenta con 5 detectores de centelleo de distintos materiales, tamaños y tipo de PMT (photomultiplier), como se ve en la siguiente tabla [2]

# ![image.png](attachment:image.png)

# En la siguiente captura se muestra el output del osciloscopio de GOLEM para cada _scintillator_

# ![image.png](attachment:image.png)

# GOLEM cuenta con condiciones favorables para la generación de _runaway electrons_, tales como una baja densidad de plasma y un alto campo eléctrico toroidal [2]. Esto lo convierte un dispositivo ideal para el estudio de este fenómeno.

# ## Análisis de señal del _scintillator_ CeBr(A)

# En los tokamaks existen dos fuentes principales de radiación por rayos X: 
# 1. Colisiones entre _runaway electrons_ y partículas residuales del plasma
# 2. Colisiones entre _runaway electrons_ y la pared
# 
# En estas colisiones el mecanismo predominante de radiación es _bremsstrahlung_.
# 
# Para detectar la aparición de _runaway electrons_ estudiamos el espectro de radiación de rayos X, el cual se puede obtener a partir de un conteo de partículas mediante _scintillators_.

# La distribución de energía de los _runaway electrons_ es la siguiente
# $$E \propto exp(-E/E_r)$$
# donde $E_r$ es la energía promedio de los _runaway electrons_ [3]. Es posible obtener esta energía realizando un análisis sobre el espectro de rayos X obtenido.

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths   
from matplotlib.ticker import AutoMinorLocator
from lmfit import Parameters,minimize, fit_report


# In[ ]:


shot_no = 50958 


# In[ ]:


ds = np.lib.npyio.DataSource('/tmp')  # temporary storage for downloaded files
data_URL = f'http://golem.fjfi.cvut.cz/shots/{shot_no}/Devices/Oscilloscopes/TektrMSO58-a/ch3.csv'
fname = ds.open(data_URL).name
df = pd.read_csv(fname, header=None, names=["time", "voltage"])
volts = df["voltage"]
time = df["time"]


# In[ ]:


## Código obtenido de rto-golem/batch_analyzer.py

import requests
import io
def load_data_from_url(url, column_names, **kwargs):
    """Carga datos desde una URL del servidor GOLEM."""
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text), header=None, names=column_names, **kwargs)
    except requests.exceptions.RequestException as e:
        print(f"Advertencia: No se pudo cargar desde {url}. Error: {e}")
        return pd.DataFrame(columns=column_names)
    
base_url = f"http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics"
ip_data = load_data_from_url(f"{base_url}/BasicDiagnostics/Results/Ip.csv", ['time_ms', 'Ip'])


# Graficamos la señal obtenida del scintillator junto a la corriente de plasma.

# In[ ]:


fig, axs = plt.subplots(2,1)

axs[0].plot(time*1000, -volts*1000)
axs[1].plot(ip_data['time_ms'], ip_data['Ip'])
axs[1].set_xlabel('Time (ms)')
axs[1].set_ylabel('Ip (kA)')
axs[0].set_ylabel('V (mV)')

axs[0].set_xlim(left=-1,right=20)
axs[1].set_xlim(left=-1,right=20)

plt.show()


# Obtenemos los peaks del voltaje (mayores al 10% del valor máximo para discriminar ruido electrónico)

# In[2]:


res = []
max_value=np.max(-volts*1e3)
threshold = 0.1 * max_value
locs, peaks = find_peaks(-volts*1e3,prominence=threshold,distance=50) 
res.append(-volts[locs]*1e3)


# Graficamos un histograma para visualizar la distribución de los peaks del voltaje
# "Pulse height distribution"

# In[ ]:


result=np.concatenate( res, axis=0 )
    
fig, ax = plt.subplots()

amp_bins = np.linspace(0,250,70)

n, edges, __ = ax.hist(result,bins=amp_bins)
centre=(edges[1:]+edges[:-1])/2

ax.set_xlabel('U [mV]')
ax.set_ylabel('N [-]')

ax.title.set_text('')
ax.set_yscale('log')
plt.show()


# (Calibración U-to-E) El voltaje U_NIM_A2 se utiliza para realizar la conversión entre voltaje y energía. Es posible obtenerlo en la base de datos de GOLEM. 

# In[ ]:


import requests
import re

url = f"http://golem.fjfi.cvut.cz/shots/{shot_no}/Production/Parameters/FullCommandLine"

try:
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Get the HTML content as a string
        html_content = response.text

        # Now you can work with the HTML content as a string
        print(html_content)
    else:
        print(f"Error: Unable to fetch the HTML content. Status code: {response.status_code}")

except requests.RequestException as e:
    print(f"Error: {e}")
# Your command line

# Regular expression pattern to match U_NIM_A2
pattern = r"U_NIM_A2=(\d+)"

# Use re.search to find the match
match = re.search(pattern, html_content)

# Check if the match is found
if match:
    # Extract the value of U_NIM_A2
    u_nim_a2_value = match.group(1)
    print(f"The value of U_NIM_A2 is: {u_nim_a2_value}")
else:
    print("U_NIM_A2 not found in the command line.")


# Convertir V a eV con la función VoltoeV. Convierte el valor medio de cada barra del histograma generado anteriormente. (Al parecer se ha calibrado de antemano el detector para conocer la conversión V a eV).
# Se descartan los intervalos del histograma con 2 conteos o menos.

# In[ ]:


a=1593.78
b=-6.19274
c=-0.384572
d=0.00572227
f=0.00111747

voldet=float(u_nim_a2_value)

x=voldet



def VoltoeV(z):
    return -(a + b*x +  d*x**2-z)/(c+f*x)

eV=VoltoeV(centre)

minus_bordel=n>2
eV=eV[minus_bordel]
n=n[minus_bordel]


# Obtenemos el conteo de partículas vs energía en eV

# In[ ]:


mask=eV>eV[np.argmax(n)]
n=n[mask]
eV=eV[mask]


# Realizamos un ajuste exponencial para obtener la energía promedio de los _runaway electrons_
# Se descartan energías menores, para considerar solo colas de alta energía (RE)

# In[ ]:


def fce(params,x,y):
        A = params['A']
        B = params['B']
        #C= params ['C']
        y_fit = A*np.exp(B*x)
        return y_fit-y
        
def fit(x,y):
    params = Parameters()

    params.add('A', min=100,max=1e15)
    params.add('B',min=-0.05,max=0)
    #params.add('C',min=0,max=0.001)

    fitted_params = minimize(fce, params, args=(x,y), method='least_squares')

    A = fitted_params.params['A'].value
    B = fitted_params.params['B'].value
    #C = fitted_params.params['C'].value    

    #print('\n---------------------lmit--------------------')
    #print(fit_report(fitted_params))
    return(A,B)


# In[ ]:


#Lower bound of the maximum runaway electron energy:

from scipy.optimize import minimize_scalar
A,B=fit(eV,n)
threshold_value=2
max_energy=np.log(threshold_value/A)/B
# Print the result
print(f"The function descends below {threshold_value} at eV = {max_energy}")
print(f"Params A:{A}, B:{B}")


# In[ ]:


# Fit para encontrar x0
from scipy.optimize import curve_fit
def exp_fit(x, A, B, x0):
    return A * np.exp(-B * (x-x0))

p0 = [np.max(n), 1, np.min(eV)]

popt, pcov = curve_fit(exp_fit, eV, n, p0=p0)

A_fit, B_fit, x0_fit = popt

print(popt)


# Graficamos la distribución de energía obtenida junto al ajuste exponencial (línea azul)

# In[ ]:


fig, ax = plt.subplots(figsize=(12,6))
FONT='Arial'
ax.set_xlabel('E [keV]')
ax.set_ylabel('N [-]')
ax.errorbar(eV,n,fmt='.',yerr=np.sqrt(n),label=r'CeBr(A),$U_{{d}}={0:.0f}$'.format(voldet),color='purple') #sets the errorbar value to square root of counts for that value

eV_new=np.linspace(eV[0],eV[-1],1000)
A,B=fit(eV,n)
ax.plot(eV_new, A*np.exp(B*eV_new), color='blue',label=r"Lower bound of the maximum $E_{{RE}}$: {0:.1f} keV".format(max_energy))



#graphic parameters
plt.xticks(fontname=FONT, fontweight = 'bold', fontsize = 15)
plt.yticks(fontname=FONT, fontweight = 'bold', fontsize = 15)

ax.tick_params(which = 'major', direction='out', length=6, width=1.5)
ax.tick_params(which = 'minor', direction='out', length=3, width=1)

for axis in ['bottom','left']:
    ax.spines[axis].set_linewidth(1.5)

ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.grid(which = 'major', c = 'gray', linewidth = 0.5, linestyle = 'solid') 
ax.grid(which = 'minor', c = 'gray', linewidth = 0.3, linestyle = 'dashed')             

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

leg = plt.legend(  loc = 'best', shadow = True, fancybox=False)

leg.get_frame().set_linewidth(1)
leg.get_frame().set_edgecolor('k')

for text in leg.get_texts():
     plt.setp(text, fontname=FONT, fontsize = 14)

ax.title.set_text('')
ax.set_ylim(bottom=0.9)
ax.set_yscale('log')
#ax.set_xscale('log')
plt.savefig('icon-fig')
plt.show()


# Como se observa en el gráfico, las energías detectadas se distribuyen aproximadamente de manera exponencial.

# Finalmente calculamos la energía promedio mediante el parámetro B y x0

# In[ ]:


Er = -1/B + x0_fit
print(Er)


# ## Futuros estudios

# En [3] se estudia cómo afectan el campo eléctrico toroidal y el campo magnético toroidal en la generación de _runaway electrons_. Se encontró que un mayor campo eléctrico incrementa la energía promedio de los electrones y a mayor campo magnético aumenta la cantidad de electrones generados. En GOLEM sería posible estudiar cómo varía el parámetro $E_r$ y el conteo de partículas modificando estos parámetros en varias descargas.

# ### Referencias
# [1] Connor, J. W., & Hastie, R. J. (1975). Relativistic limitations on runaway electrons. Nuclear fusion, 15(3), 415.  
# 
# [2] Cerovsky, et al. (2022). Progress in HXR diagnostics at GOLEM and COMPASS tokamaks. Journal of Instrumentation, 17(01), C01033.  
# 
# [3] Kafi, M., Salar Elahi, A., Ghoranneviss, M., Ghanbari, M. R., & Salem, M. K. (2016). A confident source of hard X-rays: radiation from a tokamak applicable for runaway electrons diagnosis. Synchrotron Radiation, 23(5), 1227-1231.

# ### Código Base
# https://gitlab.com/golem-tokamak/dirigent/-/blob/master/Diagnostics/ScintillationProbes/VinklarekBP22/diagnostics_runaways_JV.ipynb
