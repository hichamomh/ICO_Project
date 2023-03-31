import pandas as pd
from os import listdir
from os.path import isfile, join
fichiers = [f for f in listdir('data') if isfile(join('data', f))]
import numpy as np
import scipy 
from scipy.spatial.distance import pdist , cdist
import sklearn
from sklearn.metrics import pairwise_distances_chunked 
import matplotlib.pyplot as plt
import sklearn
from math import sin, cos, sqrt, atan2, radians
import folium
import pickle



data_depot = pd.read_csv('data_depot.csv')
data_client_index = pd.read_csv('data_clients.csv')



with open('distance_matrix.pickle', 'rb') as handle:
    distance_matrix = pickle.load(handle)

with open('times.pickle', 'rb') as handle:
    times = pickle.load(handle)



n = 50

data_depot = data_depot.iloc[:n]
data_client_index = data_client_index.iloc[:n]
distance_matrix = distance_matrix[:n , :n]
times = times[:n,:]

##############################################

V_moy = 60

def condition_valid(ti,i,j):

    """  """

    di_j = distance_matrix[i,j]
    di = data_client_index['d'].iloc[i]
    dj = data_client_index['d'].iloc[j]
    ai = data_client_index['a'].iloc[i]
    bi = data_client_index['b'].iloc[i]
    aj = data_client_index['a'].iloc[j]
    bj = data_client_index['b'].iloc[j]

    tj = ti + di + di_j/V_moy

    return int(ti >= ai and ti<= bi-di and tj >= aj and tj<= bj-dj), tj 


def condition_valid_depot(j):

    # condition de validité de 0-->j
    d0_j = data_depot['DISTANCE_KM'].iloc[j]
    dj = data_client_index['d'].iloc[j]
    aj = data_client_index['a'].iloc[j]
    bj = data_client_index['b'].iloc[j]
    tj = 480.0+dj + d0_j/V_moy

    return  int(tj >= aj and tj<= bj-dj) , tj




import random

def get_random_1_index(array,exclude_indices):
    # Trouver les indices des 1 dans le tableau, en excluant les indices donnés en argument
    ones_indices = [i for i, val in enumerate(array) if val == 1 and i not in exclude_indices ]
    
    # Vérifier s'il y a au moins un 1 dans le tableau après exclusion
    if len(ones_indices) == 0:
        return None

    # Choisir aléatoirement un indice parmi ceux des 1
    random_index = random.choice(ones_indices)

    # Déterminer l'ordre du 1 choisi
    #order = ones_indices.index(random_index) + 1
    order = sum(array[:random_index+1])
    # Retourner l'indice et l'ordre
    return random_index, order





def generate_solution(reserved_list) : 

    # Création de la matrice P0,i

    P0i = np.array([condition_valid_depot(i)[0] for i in range(data_depot.shape[0])])
    i,order = get_random_1_index(P0i,reserved_list)

    sequences = []
    solution1 = []
    matrice_solution = []

    t = condition_valid_depot(i)[1] # ti
    k = 0

    sequences.append(order)
    solution1.append(i)
    P=P0i

    matrice_solution.append(list(P))

    while True :
        
        k = k+1
        #P = np.array([int(bool(condition_valid(t,i,j)[0]) and j not in solution1) for j in range(data_depot.shape[0])])
        P = np.array([int(bool(condition_valid(t,i,j)[0])) for j in range(data_depot.shape[0])])

        if P.sum() == 0 : 
            break
        
        if get_random_1_index(P,solution1+reserved_list)==None : 

            break
        
        j,order = get_random_1_index(P,solution1+reserved_list)

        sequences.append(order)
        solution1.append(j)
        
        t = condition_valid(t,i,j)[1] # tj = ti + cij

        i = j

        matrice_solution.append(list(P))

    matrice_solution = np.array(matrice_solution).T

    
    return {'codage' : sequences , 'solution' : solution1 , 'matrice_solution' : matrice_solution}




def generate_cij(v_moy):

    C = np.zeros((distance_matrix.shape[0],distance_matrix.shape[0]))

    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[0]):
            di = data_client_index['d'].iloc[i]
            dij = distance_matrix[i][j]
            C[i,j] = di + dij/v_moy

    return C



def generate_c0j(v_moy):

    C = np.zeros((distance_matrix.shape[0],1))

    for j in range(distance_matrix.shape[0]):

            d0_j = data_depot['DISTANCE_KM'].iloc[j]
            dj = data_client_index['d'].iloc[j]
            C[j] = 480+ dj + d0_j/v_moy

    return C




Cij = generate_cij(V_moy)
C0j = generate_c0j(V_moy)


def X_to_code(X):

    matrice_X = X_to_matrice(X,distance_matrix,times).T
    codage = []

    for i in range(len(X)):
        codage.append(sum(matrice_X[i][:X[i]+1]))
        #codage.append(find_order(matrice_X[i,:],X[i]))
        
    return(codage)


def X_to_matrice(X,distances,times):

    F = [X[0]]
    col = [condition_valid_depot(i)[0] for i in range(data_depot.shape[0])]
    matrice = [col]
    t = condition_valid_depot(X[0])[1]  # ti


    for i in range(len(X)-1):
        
       
        #col = [int(bool(condition_valid(t,X[i],j)[0]) and j not in X[:i+1]) for j in range(data_depot.shape[0])]
        col = [int(bool(condition_valid(t,X[i],j)[0])) for j in range(data_depot.shape[0])]
        t = condition_valid(t,X[i],X[i+1])[1] # tj
        matrice.append(col)
        F.append(X[i+1])
    

    return(np.array(matrice).T) 


def generate_global_solution():

    reserved_list = []
    all_clients = set(data_client_index.index)
    global_solution = []

    K = 0

    while True : 

        solution = generate_solution(reserved_list)['solution']
        global_solution.append(solution)
        reserved_list = reserved_list+solution

        K = K+1
        if len(reserved_list) == len(all_clients) : 
            break

    
    return global_solution



def cout_fonction(X):

    K = len(X)
    
    cout = 0

    for Route in X : 

        c = 0
        for i in range(len(Route)-1):

            c  = c + Cij[Route[i],Route[i+1]]
        
        cout = cout+c

    return K + cout


def valid_condition(route):

    """  """
    bool = True
    T = [condition_valid_depot(route[0])[1]]

    for i in range(len(route[1:])) : 
        T.append(condition_valid(T[i-1],route[i-1],route[i])[1])
        bool = bool*condition_valid(T[i-1],route[i-1],route[i])[0]


    return bool



def get_1_indice(n,col):
    ones_indices = [i for i, val in enumerate(col) if val == 1]
    return(ones_indices[min(n,len(ones_indices))-1])


def code_to_X(code):
    # Création de la matrice P0,i

    P0i = np.array([condition_valid_depot(i)[0] for i in range(data_depot.shape[0])])
    i = get_1_indice(code[0],P0i)

     
    solution = []

    t = condition_valid_depot(i)[1] # ti
     
    solution1 = [i]
    solution.append(i)
    P=P0i
    k=1
    K = []
    indicateur = 0
    nombre_vehicule = 0
    while True :
      
        #P = np.array([int(bool(condition_valid(t,i,j)[0]) and j not in solution1) for j in range(data_depot.shape[0])])
        P = np.array([int(bool(condition_valid(t,i,j)[0] and j not in solution)) for j in range(data_depot.shape[0])])
        
        if P.sum() == 0 :
            #print(len(solution))
            if  len(solution)== len(code):
                K.append(solution1)
                solution1 = [] 
                break
                
            
            else:
                P = np.array([int(condition_valid_depot(i)[0] and i not in solution) for i in range(data_depot.shape[0])])
                # modif
                
                #print('taboun')
                
                # modif
                K.append(solution1)
                solution1 = [] 
                indicateur = 1
                
                
        #print(P)
        j = get_1_indice(code[k],P)
        solution.append(j)
        solution1.append(j)
        
        if indicateur==1 : 
            t = condition_valid_depot(j)[1]
            indicateur = 0
            
        else : 
            
            t = condition_valid(t,i,j)[1] # tj = ti + cij
            
        i = j
        k+=1


    return K