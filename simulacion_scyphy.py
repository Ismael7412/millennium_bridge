import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from numba import njit
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from matplotlib.colors import Normalize

@njit
def sistema(t, X, omega, omega_0, G, M, B, K, C, alpha):
    
    #X es un array con todas las variables del sistema
    
    x = X[0]
    V = X[1]
    theta = X[2:]
    
    #defino las listas y variables necesarias y que son dependientes del numero de personas (N)
   
    A = np.sqrt(x**2 + (V/omega_0)**2)
    Psi = np.arctan2(x, V/omega_0)
        
    #definimos el sistema de ecuaciones diferenciales acopladas
    dx_dt = V
    dV_dt = (G/M)*np.sum(np.sin(theta))-B*V/M-K*x/M
    dtheta_dt = omega + C*A*np.sin(Psi - theta + alpha)
        
        
    #preparamos una lista que devolverá la X,V y las thetas
    return_list = [dx_dt, dV_dt]

    return return_list + list(dtheta_dt)
     

aux_complex = 1j
#CONSTANTES DEL PUENTE

M = 1.13e5 #kg
B = 1.10e4 #kg/s
K = 4.73e6 #kg/s^2
G = 30 #kg*m/s^2
C = 16 #m^-1 s^-1
alpha = np.pi/2 #rad
omega_0 = np.sqrt(K/M) #rad/s
std_omega = 0.63 #rad/s

    
#CONSTANTES DEL ALGORITMO

t_max = 2000                 #tiempo considerado
t = np.arange(0, t_max-1, 1)   #array con todos los tiempos
Nt = np.zeros(t_max-1)         #array tal que N[t] es el numero de personas en t

t_start_50 = 250 #(s)        #añadimos las 50 personas en t=250
t_stop_50 = 500 #(s)         #empiezan los incrementos de 10 personas en t=500
t_step = 100 #(s)            #el tiempo entre incrementos de 10 personas
t_final = t_max-200 #(s)     #tiempo q dejamos al final de la grafica con N=cte



#llenamos el array Nt desde t_start_50 a t_stop_50 con 50 personas
for i in range(t_stop_50-t_start_50):
    Nt[i+t_start_50] = 50

#llenamos Nt con todas las N en cada uno de los t. Sumando diez en cada paso
for i in range(int((t_max-t_stop_50)/t_step)-1):
    for j in range(t_step):
        Nt[t_stop_50+i*t_step+j] += (i+1)*10 + 50
        
#dejamos como en el paper un numero cte de N los ultimos segundos
for i in range(t_final, t_max-1):
    Nt[i] = 50 + ((t_final-t_stop_50)/t_step+1)*10
        

N_max = int(Nt[-1])
#array con los resultados a plotear
X_results = []
R_results = []
V_results = []


#***** REGION 1: N=0 desde 0 a t_start_50
#en esta región no hay ni gente ni balanceo, metemos ceros en resultados

X_results.extend(np.zeros(t_start_50-1))
V_results.extend(np.zeros(t_start_50-1))
R_results.extend(np.zeros(t_start_50-1))
tetha_results=np.zeros((N_max, t_max))


#***** REGION 2: N=50 desde t_start_50 a t_stop_50

N=50
initial_conditions = 2*np.pi*np.random.rand(N+2)
initial_conditions[0] = 0
initial_conditions[1] = 0       #puente en reposo

#inicializamos las omegas de cada peaton
omega = np.random.normal(omega_0, std_omega, N)

#resolvemos la ecuación diferencial
args = [omega, omega_0, G, M, B, K, C, alpha]
t_eval = np.arange(t_start_50, t_stop_50, 1)
sol = solve_ivp(sistema, (t_start_50,t_stop_50-1), initial_conditions, t_eval=t_eval, args=args)

X_results.extend(sol.y[0])
V_results.extend(sol.y[1])
for k in range(N_max):
    if k < N:
        tetha_results[k][t_start_50:t_stop_50] = sol.y[k+2]

#calculamos el parámetro de orden en cada t
for tt in range(len(sol.y[0])):
    aux = 0.0
    for i in range(N):
        aux += np.exp(sol.y[i+2][tt]*aux_complex)
    R_results.append(abs(aux/N))
    
    
    
#***** REGION 3: Aumentamos N en 10 en cada paso de 100s

for i in range(int((t_final-t_stop_50)/t_step)):
    
    #numero de personas en el paso i
    N = int(50 + 10*(i+1))
    
    #inicializamos el estado inicial con el ultimo valor de cada variable del paso anterior
    initial_conditions = np.concatenate((initial_conditions, 2*np.pi*np.random.rand(10)))
    initial_conditions[0] = sol.y[0][-1]
    initial_conditions[1] = sol.y[1][-1]
    for k in range(int(N-10)):
        initial_conditions[k+2] = sol.y[k+2][-1]
    
    #añadimos nuevas frecuencias asociadas a los nuevos peatones
    omega = np.concatenate((omega, np.random.normal(omega_0, std_omega, 10)))
    args = [omega, omega_0, G, M, B, K, C, alpha]
    
    t_eval = np.arange(t_stop_50+i*t_step, t_stop_50+i*t_step+t_step, 1)
    sol = solve_ivp(sistema, (t_stop_50+i*t_step, t_stop_50+i*t_step+t_step-1), initial_conditions, t_eval=t_eval, args=args)
        
    X_results = np.concatenate((X_results, sol.y[0]))
    V_results = np.concatenate((V_results, sol.y[1]))
    for k in range(N_max):
        if k < N:
            tetha_results[k][t_stop_50+i*t_step:t_stop_50+i*t_step+t_step] = sol.y[k+2]
    
    #calculamos el parámetro de orden en cada t
    for tt in range(len(sol.y[0])):
        aux = 0.0
        for i in range(N):
            aux += np.exp(sol.y[i+2][tt]*aux_complex)
        R_results.append(abs(aux/N))
        
        
        
# ***** REGION 4: Mantenemos N constante desde t_final a t_max

#añadimos las ultimas diez personas
N += 10
omega = np.concatenate((omega, np.random.normal(omega_0, std_omega, 10)))
args = [omega, omega_0, G, M, B, K, C, alpha]

#inicializamos por ultima vez las condiciones iniciales
initial_conditions = np.concatenate((initial_conditions, 2*np.pi*np.random.rand(10)))
initial_conditions[0] = sol.y[0][-1]
initial_conditions[1] = sol.y[1][-1]
for i in range(int(N-10)):
    initial_conditions[i+2] = sol.y[i+2][-1]

#ultima integracion
t_eval = np.arange(t_final, t_max, 1)
sol = solve_ivp(sistema, (t_final, t_max-1), initial_conditions, t_eval=t_eval, args=args)

X_results = np.concatenate((X_results, sol.y[0]))
V_results = np.concatenate((V_results, sol.y[1]))
for k in range(N_max):
     if k < N:
         tetha_results[k][t_final:t_max] = sol.y[k+2]


#calculamos el parámetro de orden en cada t
for tt in range(len(sol.y[0])):
    aux = 0.0
    for i in range(N):
        aux += np.exp(sol.y[i+2][tt]*aux_complex)
    R_results.append(abs(aux/N))
    

#********* RESULTADOS

A = np.sqrt(X_results**2+(V_results/omega_0)**2)

#Creamos un dataframe para visualizar todos los resultados
df_data = {"Tiempo (s)": t, "Personas": Nt, "Amplitud de oscilacion": A*100, "Parametro de orden": R_results}
df = pd.DataFrame(df_data)

axes = df.plot(0, [1, 2, 3],subplots=True, grid=True, figsize=(8, 6), legend=False)
axes[0].set_ylabel('Crowd Size')
axes[1].set_ylabel('Wobbling Amplitude (cm)')
axes[2].set_ylabel('Order Parameter')

tetha_results_sin = np.sin(tetha_results)




# # ********** GIF ANIMADO

# t = np.append(t, 1999)

# # Set up the initial plot
# fig, ax = plt.subplots()
# im = ax.imshow(np.zeros((190, 2000)), cmap='coolwarm', aspect='auto', norm=Normalize(vmin=-1, vmax=1))

# ax.set_xlabel('t (s)')
# ax.set_ylabel('sin(θᵢ(t))')

# # Define the update function
# def update(frame):
#     y = tetha_results_sin[:]  # Select the current frame of theta
#     im.set_data(y)
#     if 4*frame > 1900:
#         frame=1900/4
#     ax.set_xlim(0, 100+4*frame)  # Set x-axis limits based on the length of t
#     ax.set_ylim(0, 190)  # Set y-axis limits based on the current number of lists
#     return im,

# # Create the animation
# frames = int((2000-100)/3)  # Total number of frames (excluding the first 100)
# ani = FuncAnimation(fig, update, frames=frames, interval=0, blit=True)

# # Show the animation
# plt.show()










