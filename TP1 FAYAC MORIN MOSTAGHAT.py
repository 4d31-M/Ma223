import numpy as np
import random
import time
import matplotlib.pyplot as plt




def ReductionGauss(Aaug):
    n,m = np.shape(Aaug)

    for i in range (0,n-1):
        for j in range (i+1, n):
            a = Aaug[j,i]
            for l in range (i, m):
                Aaug[j,l] = Aaug[j,l] - (a/Aaug[i,i]) * Aaug[i,l]
    return Aaug

def ResolutionSystTriSup(Taug): 
    n,m=np.shape(Taug)
    X=np.zeros((n,1))
    X[n-1,0]=Taug[n-1,n]/Taug[n-1,n-1]
    for i in range(n-2,-1,-1):
        X[i]=Taug[i,n]
        for j in range(i+1,n):
            X[i]=X[i]-Taug[i,j]*X[j]
        X[i] = (1/Taug[i,i])*X[i]
    return(X)

def Gauss(A,B):
    
    Aaug = np.concatenate((A,B),axis=1)

    
    Taug = ReductionGauss(Aaug)

    
    F = ResolutionSystTriSup(Taug)
    
    return F




def DecompositionLU(A):

    n,m = np.shape(A)

    U = np.copy(A)
    L = np.eye(n)
    
    
    for i in range(0,n-1):
        
        for j in range(i+1,n):
            p = U[j,i]/U[i,i]
            L[j,i]= p
            
            for k in range(i,n):
                U[j,k]= U[j,k]-p*U[i,k]
    
    return L,U

def ResolutionLU (L,U,B):
    n,m = np.shape(L)
    y=[]
    x=np.zeros((m,1))
    y.append(B[0,0])

    for i in range (0,n-1):
        G=0
        for k in range (-2,i-1):
            G = G+L[i+1,k+2]*y[k+2]
        y.append(B[i+1,0]-G)
        
    for i in range(n-1,-1,-1):
        H=0
        for j in range(i+1,n):
            H = H+U[i,j]*x[j]
        x[i,0] = (1/U[i,i])*(y[i]-H)    
    return x

def lu(A,B):
    
    [L,U] = DecompositionLU(A)
    y=ResolutionLU(L,U,B)
    return y







def ReductionGauss_partiel(Aaug):
    
    n,m=Aaug.shape
    
    for i in range (n-1):
        imax = i
        for k in range(i+1, n):                              
            if (abs(Aaug[k][i]) > abs(Aaug[imax][i])):
                imax = k

        if (imax != i):
            Aaug[[imax,i]]=Aaug[[i,imax]]
        pivot=Aaug[i,i]
        
        if (pivot==0):
            print("Malheureusement le pivot est nul, le programme ne pourra pas\ns'exécuter car la matrice donnée est susceptible d'être non inversible.")
        
        elif (pivot!=0):
            
            for j in range(i+1,n):
                a = Aaug[j,i]
                for l in range(i,m):
                    Aaug[j,l]=Aaug[j,l]-(a/pivot)*Aaug[i,l]

    return(Aaug)


def GaussChoixPivotPartiel(A,B):    
    n,m=A.shape
    for i in range (n):
        Aaug=np.append(A,B,axis=1)
        
    Taug=ReductionGauss_partiel(Aaug)
    
    C=ResolutionSystTriSup(Taug)
    
    
    return (C)






def main():
    n = 20
    
    
    L_heures = []
    l_time = []
    L_temps = []
    nb_matrice = []
    taille = []
    Lsize = []
    a = 0
    indice = a
    ListeNorme = []
    LNorm = []
    L_err = []
    for i in range (1,n,2):

        a = i
        print(i)
        A = np.random.rand(a, a)
        B = np.random.rand(a, 1)
        
        #Gauss
        start_time = time.time()
        X = Gauss(A,B)
        time_f = time.time() - start_time
        
        l_time.append(time_f)
        nb_matrice.append(i)
        AXB=np.dot(A,np.ravel(X))-np.ravel(B)
        N=np.linalg.norm(AXB)
        ListeNorme.append(N)
        
        #LU
        time_dep = time.time()
        X = lu(A,B)
        fin = time.time() - time_dep

        L_temps.append(fin)
        taille.append(i)
        AXB=np.dot(A,np.ravel(X))-np.ravel(B)
        N=np.linalg.norm(AXB)
        LNorm.append(N)

        #Partiel
        dep = time.time()
        X = GaussChoixPivotPartiel(A,B)
        termine = time.time() - dep

        L_heures.append(termine)
        Lsize.append(i)
        AXB=np.dot(A,np.ravel(X))-np.ravel(B)
        N=np.linalg.norm(AXB)
        L_err.append(N)
        
        




    total_time = 0
    for i in range(len(l_time)) :
        total_time += l_time[i]
        #print(l_time[i])
    
    tot = 0
    for i in range(len(L_temps)):
        tot +=  L_temps[i]
        #print(L_temps[i])

    entier = 0
    for i in range(len(L_heures)):
        entier += L_heures[i]
        #print(L_heures[i])
    
    print("Temps total en secondes : ",total_time)
    print("Temps par matrice: ", total_time/len(l_time))
    
    titre = "Temps de calcul pour des matrices de tailles croissantes jusqu'à ",int(indice),". "
    fig, ax = plt.subplots()
    ax.plot(nb_matrice, l_time, "r-*", color = "r",label=r"$Gauss$")
    ax.plot(taille, L_temps, "r-*", color = "b",label = r"$LU$")
    ax.plot(Lsize, L_heures, "r-*", color = "g",label = r"$Pivot partiel$")
    plt.legend()
    plt.grid(True)
    plt.title(titre)
    plt.xlabel("Tailles des matrices")
    plt.ylabel("Temps en secondes par matrice")

    
    plt.show()
    plt.close()





    plt.grid(True)
    plt.plot(nb_matrice,ListeNorme,"r-*",color="r",label=r"$Gauss$")
    plt.legend()
    fig, ax = plt.subplots()
    plt.grid(True)
    plt.plot(taille,LNorm,"r-*",color="b",label = r"$LU$")
    plt.legend()
    fig, ax = plt.subplots()
    plt.grid(True)
    plt.plot(Lsize,L_err,"r-*",color="g",label = r"$Pivot partiel$")
    
    plt.legend()
    plt.xlabel("Taille de la matrice A")
    plt.ylabel("Erreurs")
    plt.title("Erreur sur la méthode de Gauss en fonction de la taille de la matrice A")
    plt.savefig("norme gauss LU")
    plt.show()
    plt.close()







if __name__ == '__main__':
    main()

