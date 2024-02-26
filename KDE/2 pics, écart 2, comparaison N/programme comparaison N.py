import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

# générer l'échantillon à partir de deux lois normales : 
N = 1000 # mettre le nombre de points voulu
h = 0.25 # valeur de la bandwidth

X = np.concatenate(
    (np.random.normal(0, 1, int(0.4 * N)), np.random.normal(5, 2, int(0.6 * N))))[:, np.newaxis]

# préparer les points où on calculera la densité
X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

# préparation de l'affichage de la vraie densité, qui est celle à partir de laquelle les données 
#ont été générées (voir plus haut)
#la pondération des lois dans la somme est la pondération des lois dans l'échantillon généré (voir plus haut)
true_dens = 0.4 * norm(0, 1).pdf(X_plot[:, 0]) + 0.6 * norm(5, 2).pdf(X_plot[:, 0])

# estimation de densité par noyaux gaussiens
kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(X)

# calcul de la densité pour les données de X_plot : 
density = np.exp(kde.score_samples(X_plot))

# affichage : vraie densité et estimation : 
fig = plt.figure()
ax = fig.add_subplot(111) #format du graphique
ax.fill(X_plot[:,0], true_dens, fc='black', alpha=0.2, label='distribution en entrée')
ax.plot(X_plot[:,0], density, '-', label="Estimation")
ax.legend(loc='upper left')

#affichage du nombre de points de la mesure : 
ax.text(6, 0.38, "N={0} points".format(N))
ax.text(6, 0.35, "h={0}".format(h))

#afficher la distribution des points en bas du graphique : afficher que pour N=10 ou N=100 sinon trop chargé
ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), "+k")

#limites du graphique : 
ax.set_xlim(-4, 10)
ax.set_ylim(-0.02, 0.4)

#Ajout de la légende : 
ax.set_xlabel("x")
ax.set_ylabel("Densité normalisée")

#sauvergarde du fichier : 
plt.savefig('2 pics (écart 2 sur le pic 2), h=0.25, N=1000.png', dpi=500)