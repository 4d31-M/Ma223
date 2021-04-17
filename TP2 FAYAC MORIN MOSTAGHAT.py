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

def ResolutionSystTriInf(Taug):
    n,m=np.shape(Taug)
    x=np.zeros((n,1))
    for k in range(n):
        somme =0
        for i in range(k):
            somme=somme+ Taug[k,i]*x[i,0]
        x[k,0]=(Taug[k,-1]-somme)/Taug[k,k]
    return x

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



def cholesky(A) :
    At = A.transpose()
    if At.all() != A.all() :
        print("La matrcice n'est pas symétrique !\nAlgorithme impossible !")
        return
    try :
        n,m = np.shape(A)
        L = np.zeros((n,n))
        for k in range(0,n):
            somme = 0
            for j in range(0,k):
                somme = somme + L[k,j]**2
            L[k,k] = (A[k,k] - somme)**(1/2)
            for i in range(k+1,n):
                somme = 0
                for j in range(k):
                    somme = somme + L[i,j] * L[k,j]
                L[i,k] = (A[i,k] - somme) / L[k,k]
    except :
        print("Matrice non définie positive donc Cholesky n'est pas possible.")
    return L

def ResolCholesky(A,B):
    n,m=np.shape(A)
    L=cholesky(A)
    Laug = np.concatenate((L,B),axis=1)
    Y = np.zeros(n)
    Y = ResolutionSystTriInf(Laug)
    Yaug = np.concatenate((L.T,Y),axis=1)
    X = ResolutionSystTriSup(Yaug)
    return X

def symetrique(n):
    n = int(n)
    M = np.zeros((n,n))
    for i in range(0,n):
        for j in range (i,n) :
            a = random.random()
            M[i,j] = a
            M[j,i] = a
    Mt = M.T

    A = np.dot(M,Mt)
    
    return A
        



def main():
    n = 500
    
    
    L_heures = []
    l_time = []
    L_temps = []
    cholesky_temps = []
    nb_matrice = []
    taille = []
    Lsize = []
    cholesky_taille = []
    a = 0
    indice = a
    ListeNorme = []
    LNorm = []
    L_err = []
    ch_norme = []
    print("Resolution en cours...")
    for i in range (1,n, 41):

        a = i
        print("matrice {}".format(i))
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
        
        #Cholesky
        A = symetrique(a)
        ch_time = time.time()
        X = ResolCholesky(A,B)
        time_résolutionch = time.time() - ch_time
        
        cholesky_temps.append(time_résolutionch)
        cholesky_taille.append(i)
        AXB=np.dot(A,np.ravel(X))-np.ravel(B)
        N=np.linalg.norm(AXB)
        ch_norme.append(N)




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

    totalch = 0
    for i in range(len(cholesky_temps)):
        totalch += cholesky_temps[i]
    
    print("Temps total en secondes : ",total_time)
    print("Temps par matrice: ", total_time/len(l_time))
    
    #echelle linéaire
    fig, ax = plt.subplots()
    ax.plot(nb_matrice, l_time, "r-", color = "r",label=r"$Gauss$")
    ax.plot(taille, L_temps, "r-", color = "b",label = r"$LU$")
    ax.plot(Lsize, L_heures, "r-", color = "g",label = r"$Pivot partiel$")
    ax.plot(cholesky_taille, cholesky_temps, "r-", color = "c",label = r"$Cholesky$")
    plt.legend()
    plt.grid(True)
    plt.title("Temps de calcul, matrices de tailles croissantes (taille max :{})".format(n))
    plt.xlabel("Tailles des matrices")
    plt.ylabel("Temps en secondes par matrice")

    #echelle logarithmique
    fig, ax = plt.subplots()
    ax.plot(nb_matrice, l_time, "r-", color = "r",label=r"$Gauss$")
    ax.plot(taille, L_temps, "r-", color = "b",label = r"$LU$")
    ax.plot(Lsize, L_heures, "r-", color = "g",label = r"$Pivot partiel$")
    ax.plot(cholesky_taille, cholesky_temps, "r-", color = "c",label = r"$Cholesky$")
    plt.legend()
    plt.grid(True)
    plt.title("log Temps de calcul, matrices de tailles croissantes (taille max :{})".format(n))
    plt.xlabel("Tailles des matrices")
    plt.ylabel("Temps en secondes par matrice")
    plt.yscale('log')
    plt.xscale('log')

    plt.show()
    plt.close()


    #echelle linéaire
    plt.grid(True)
    plt.plot(nb_matrice,ListeNorme,"r-",color="r",label=r"$Gauss$")
    plt.xlabel("Taille de la matrice A")
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("Erreurs")
    plt.title("Erreur en fonction de la taille de la matrice (Gauss)")
    plt.legend()
    fig, ax = plt.subplots()
    plt.grid(True)
    plt.plot(taille,LNorm,"r-",color="b",label = r"$LU$")
    plt.xlabel("Taille de la matrice A")
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("Erreurs")
    plt.title("Erreur en fonction de la taille de la matrice (LU)")
    plt.legend()
    fig, ax = plt.subplots()
    plt.grid(True)
    plt.plot(Lsize,L_err,"r-",color="g",label = r"$Pivot partiel$")
    plt.xlabel("Taille de la matrice A")
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("Erreurs")
    plt.title("Erreur en fonction de la taille de la matrice (Pivot partiel)")
    plt.legend()
    fig, ax = plt.subplots()
    plt.grid(True)
    plt.plot(cholesky_taille,ch_norme,"r-",color="c",label=r"$Cholesky$")
    plt.legend()
    plt.xlabel("Taille de la matrice A")
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("Erreurs")
    plt.title("Erreur en fonction de la taille de la matrice (Cholesky)")
    plt.savefig("norme gauss LU")


    #echelle logarithmique
    fig2, ax2 = plt.subplots()
    ax2.plot(nb_matrice,ListeNorme,"r-",color="r",label=r"$Gauss$")
    ax2.plot(taille,LNorm,"r-",color="b",label = r"$LU$")
    ax2.plot(Lsize,L_err,"r-",color="g",label = r"$Pivot partiel$")
    ax2.plot(cholesky_taille,ch_norme,"r-",color="c",label=r"$Cholesky$")
    plt.legend()
    plt.grid(True)
    plt.xlabel("Taille de la matrice A")
    plt.ylabel("Erreurs")
    plt.title("logErreur en fonction de la taille de la matrice")
    plt.yscale('log')
    plt.xscale('log')
    
    fig3, ax3 = plt.subplots()
    ax3.plot(nb_matrice,ListeNorme,"r-",color="r",label=r"$Gauss$")
    ax3.plot(taille,LNorm,"r-",color="b",label = r"$LU$")
    ax3.plot(Lsize,L_err,"r-",color="g",label = r"$Pivot partiel$")
    ax3.plot(cholesky_taille,ch_norme,"r-",color="c",label=r"$Cholesky$")
    plt.legend()
    plt.grid(True)
    plt.xlabel("Taille de la matrice A")
    plt.ylabel("Erreurs")
    plt.title("Erreur en fonction de la taille de la matrice")
    
    plt.show()
    plt.close()




if __name__ == '__main__':
    main()

