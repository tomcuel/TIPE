# Importation des bibliothèques utiles à notre traitement de données 
from math import * 

# Numpy pour les matrices
import numpy as np

# MatplotLib pour les graphiques
import matplotlib.pyplot as plt

# Scipy Stats pour réaliser des estimations par densité de noyau
from scipy.stats import kde

# Un module annexe d'estimation par noyau pour régler une contrainte
from statsmodels.nonparametric.kde import KDEUnivariate

# Un module utile pour les représentations doubles des estimations par densité de noyau
import seaborn as sns

# Lecture des données issues des relevés GPS sous forme .txt
# Avec la création de différents tableaux de données
# Calcul des intervalles de distances, des intervalles de temps, et autres


def LectureApp(string):
    
    # Lecture du document
    texte  = open(string, "r")
    t = texte.readlines()
    a = ""
    
    # On récupère le nombre de lignes
    for i in range(len(t)):
        a += t[i]
    
    # Ensuite, avec toutes les fonctions suivantes, on va extraire les 
    # données qui nous intéressent.
    
    #fonction qui cherche si le 'motif' est à la position 'i' dans le 'texte'
    def est_ici(texte,motif,i):
        a = 0
        if len(motif)>len(texte):
            return False
    
        while a<len(motif):
            if texte[i+a] == motif[a]:
                a += 1
            else:
                return False
        return True
    
    #fonction qui cherche si le 'motif' est dans le 'texte'
    def est_sous_mot(texte,motif):
        indice = 0
        while indice < len(texte):
            if est_ici(texte,motif,indice) == False :
                indice += 1
            else:
                return True
        return False
    
    #fonction qui cherche la(les) position(s) du motif 'i' dans le 'texte'
    def position_sous_mot(texte,motif):
        occurence = []
        for i in range (len(texte)):
            if est_ici(texte, motif, i) ==True :
                occurence.append(i)
        return(occurence)
    
    #fonction qui donne la liste contenant les différentes positions des motifs 
    # 'c' et 'b' dans le texte
    def liste_pour_recherche_de_mot(c,b):
        D = position_sous_mot(a,c) #position(s) du motif 'c' dans le texte 'a'
        E = position_sous_mot(a,b) #position(s) du motif 'b' dans le texte 'a'
        n = len(c)
        for i in range(len(D)):
            D[i] += n
        return D,E
    
    liste = liste_pour_recherche_de_mot("<DistanceMeters>","</DistanceMeters>")
    liste2 = liste_pour_recherche_de_mot("<Time>","</Time")
    
    def entre_occurence (l1,l2):
        L = []
        for i in range(len(l1)):
            longueur = l2[i]-l1[i]
            mot = ""
            for j in range(longueur):
                mot += a[l1[i]+j]
            L.append(mot)
        return L
    
    def intervalle_temps(liste):
        L = []
        for i in range(len(liste)-1):
            donnees1 = liste[i].split('T')
            donnees2 = liste[i+1].split('T')
            donnees3 = donnees1[1].split(".")
            donnees4 = donnees2[1].split(".")
            donnees5 = donnees3[0].split(":")
            donnees6 = donnees4[0].split(":")
            if donnees1[0] == donnees2[0]:
                L.append((int(donnees6[0])*3600+int(donnees6[1])*60+int(donnees6[2]))-(int(donnees5[0])*3600+int(donnees5[1])*60+int(donnees5[2])))
            else:
                L.append((int(donnees6[0])*3600+int(donnees6[1])*60+int(donnees6[2])+3600*24)-(int(donnees5[0])*3600+int(donnees5[1])*60+int(donnees5[2])))
        return L
    
    def intervalle_distance(liste):
        L = []
        for i in range(1,len(liste)-1):
            L.append(float(liste[i+1])-float(liste[i]))
        return L
    
    intervale_distance = intervalle_distance(entre_occurence(liste[0],liste[1]))
    intervale_temps = intervalle_temps(entre_occurence(liste2[0],liste2[1]))
    
    p = []
    m = []
    somme = 0
    somme_bis = 0
    temoin = []
    
    #condition : ne les prendra pas en compte dans les expériences statiques :
    
    #expérience de 60min : 
    if len(intervale_distance)==1037 or len(intervale_distance)==1269 or len(intervale_distance)==1576 :
        vmoy = 5.2390465693 
        for i in range(len(intervale_temps)):
            somme += intervale_distance[i]
            somme_bis += intervale_temps[i]
            p.append(somme)
            m.append(somme_bis)
            if abs(somme-(vmoy*somme_bis)) < 1000 :
                temoin.append(abs(somme-(vmoy*somme_bis)))
            if abs(somme-(vmoy*somme_bis)) > 1000 :
                m.pop()
                p.pop()
     
    #expérience de 40min :     
    elif len(intervale_distance)==674 or len(intervale_distance)==775 or len(intervale_distance)==1121 :
        vmoy = 5.1399026764
        for i in range(len(intervale_temps)):
            somme += intervale_distance[i]
            somme_bis += intervale_temps[i]
            p.append(somme)
            m.append(somme_bis)
            if abs(somme-(vmoy*somme_bis)) < 1000 :
                temoin.append(abs(somme-(vmoy*somme_bis)))
            if abs(somme-(vmoy*somme_bis)) > 1000 :
                m.pop()
                p.pop()
    #expérience de 20 min :      
    elif len(intervale_distance)==336 or len(intervale_distance)==426 or len(intervale_distance)==518 :
        vmoy = 5.25497512438
        for i in range(len(intervale_temps)):
            somme += intervale_distance[i]
            somme_bis += intervale_temps[i]
            p.append(somme)
            m.append(somme_bis)
            if abs(somme-(vmoy*somme_bis)) < 1000 :
                temoin.append(abs(somme-(vmoy*somme_bis)))
            if abs(somme-(vmoy*somme_bis)) > 1000 :
                m.pop()
                p.pop()
   
    #expérience statique : 
    else:
        for i in range(len(intervale_distance)):
            somme += intervale_distance[i]
            somme_bis += intervale_temps[i]
            p.append(somme)
            m.append(somme_bis)

        
    # On met dans une nouvelle liste les données suivantes sous forme de listes :
    K=[]
    # temps total
    K.append(m)
    # distance totale
    K.append(p)
    # intervalle temps
    K.append(intervale_temps)
    # intervalle distance
    K.append(intervale_distance)
    # moyenne de la distance parcourue par seconde
    K.append(temoin)
    
    return(K)



# Commande pour obtenir les données relatives à l'expérience dynamique 
# de 20 minutes pour le capteur Garmin 735 XT (montre) :
    
# print(LectureApp("ACT-BELDYN-20m-GARMIN735.txt"))



def mesure_des_ecarts_statique():
    
    #Indication dans la console de quelle partie il s'agit
    print("MESURE DES ÉCARTS STATIQUES \n")
    
    #Récupération des listes de données des différents fichiers.
    A = LectureApp("activity_extérieur 1_GPS_vélo.txt")
    B = LectureApp("activity_extérieur 2_GPS_vélo.txt")
    C = LectureApp("activity_intérieur 1_GPS_vélo.txt")
    D = LectureApp("activity_intérieur 2_GPS_vélo.txt")
    
    if len(C[2]) > len(C[3]) : 
        while len(C[2]) != len(C[3]) :
            C[2].pop()
    if len(C[3]) > len(C[2]) : 
        while len(C[2]) != len(C[3]) :
            C[3].pop()
    
    Z=LectureApp("activity_extérieur 1_GPS_vélo.txt")
    for i in range (len(Z[0])) :
        Z[0][i] = Z[0][i]/60
    for i in range (len(Z[1])) :
        Z[1][i] = Z[1][i]
            
    Y=LectureApp("activity_extérieur 2_GPS_vélo.txt")
    for i in range (len(Y[0])) :
        Y[0][i] = Y[0][i]/60
    for i in range (len(Y[1])) :
        Y[1][i] = Y[1][i]
    
    X=LectureApp("activity_intérieur 1_GPS_vélo.txt")
    if len(X[2]) > len(X[3]) : 
        while len(X[2]) != len(X[3]) :
            X[2].pop()
    if len(X[3]) > len(X[2]) : 
        while len(X[2]) != len(X[3]) :
            X[3].pop()
    for i in range (len(X[0])) :
        X[0][i] = X[0][i]/60
    for i in range (len(X[1])) :
        X[1][i] = X[1][i]
        
    W=LectureApp("activity_intérieur 2_GPS_vélo.txt")
    for i in range (len(W[0])) :
        W[0][i] = W[0][i]/60
    for i in range (len(W[1])) :
        W[1][i] = W[1][i]

    # Informations utiles
    
    print(" -------- Informations Utiles -------- \n")
    print("Nombre de mesures (1h30 en extérieur) : "+str(len(A[0])))
    moycoef = (coeff(A)+coeff(B)+coeff(C)+coeff(D))/4
    print("Moyenne des coefficients : " + str(moycoef))
    
    # Affichage de l'estimation de densité par noyau (KDE) des intervalles de temps
    
    # Estimation de densité par noyau (via fonction auxiliaire)
    AK = GAUSSIENNE(A[2])
    BK = GAUSSIENNE(B[2])
    CK = GAUSSIENNE(C[2])
    DK = GAUSSIENNE(D[2])
    
    # Affichage
    plt.clf()
    plt.plot(AK[0], AK[1], color="blue")
    plt.plot(BK[0], BK[1], color="red")
    plt.plot(CK[0], CK[1], color="green")
    plt.plot(DK[0], DK[1], color="maroon")
    plt.title("Densité statistique des intervalles de temps mesurés")
    plt.xlabel("Intervalle de temps (en secondes)")
    plt.ylabel("Probabilité effective (0.1 = 10%)")
    plt.savefig("ACT-GAUSSIENNE-TEMPS-GPS-vélo.png", dpi=500)
    
    #Affichage des régressions linéaires avec la moyenne
    plt.clf()
    
    # Calcul des valeurs moyennes
    mh=moycoef*60 #en m/min
    mhh=moycoef*3600 #en m/h
    time=[]
    long=[]
    for i in range(1000):
        time.append(i*200/1000)    
        long.append(i*200*mh/1000)
    mhh=(int(str(mhh*10)[:2]))/10 # en m/h
    plt.plot(time, long, color='black', label="moyenne, a="+str(mhh)+"m/h")
    
    # Mise en forme
    plt.xlabel("Temps (en min)")
    plt.ylabel("Distance (en mètres)")
    plt.title("régression linéaire moyenne")
    plt.legend(loc=2)
    plt.savefig("ACT-Courbes données brutes GPS vélo.png", dpi=500)

    return(None)
    
    
    
    
# Définition de plusieurs fonctions Annexes utiles pour nos fonctions principales


# Calcul de coefficient par régression linéaire
def coeff(L):
    tps = L[0]
    parc= L[1]
    coef, ordo= np.polyfit(tps, parc, 1)
    return(coef)


# Construction de deux listes pour exploiter la régression linéaire dans des graphiques
def plotage(L):
    coef = coeff(L)
    t=[]
    mg=[]
    for i in (L[0]):
        t.append(i)
        mg.append(i*coef)
    K=[]
    K.append(t)
    K.append(mg)
    return(K)


# Première fonction d'estimation de densité par noyau gaussien utilisée
def GAUSSIENNE(b):
    density = kde.gaussian_kde(b)
    x = np.linspace(-5,45, 400)
    y=density(x)
    return(x,y)


# Deuxième fonction d'estimation de densité par noyau gaussien utilisée
def GAUSSIENNE2(b):
    density = kde.gaussian_kde(b)
    x = np.linspace(-2,8, 400)
    y=density(x)
    return(x,y)

# Fonction annexe du calcul de KDE (car on a rencontré quelques problèmes avec le module de base)
def kde_statsmodels_u(x, x_grid, bandwidth=3, **kwargs):
    kde = KDEUnivariate(x)
    kde.fit(bw=bandwidth, **kwargs)
    return kde.evaluate(x_grid)




# FONCTION DE LANCEMENT DU PROGRAMME 

def Start():
    mesure_des_ecarts_statique()

Start()