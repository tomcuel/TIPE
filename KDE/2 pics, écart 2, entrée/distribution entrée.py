import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# générer l'échantillon à partir de deux lois normales : 
N = 50 # mettre le nombre de points voulu

X = np.concatenate(
    (np.random.normal(0, 1, int(0.35 * N)), np.random.normal(5, 2, int(0.65 * N))))[:, np.newaxis]

# préparer les points où on calculera la densité
X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

# préparation de l'affichage de la vraie densité, qui est celle à partir de laquelle les données 
#ont été générées (voir plus haut)
#la pondération des lois dans la somme est la pondération des lois dans l'échantillon généré (voir plus haut)
true_dens = 0.4 * norm(0, 1).pdf(X_plot[:, 0]) + 0.6 * norm(5, 2).pdf(X_plot[:, 0])

# affichage : vraie densité et estimation : 
fig = plt.figure()
ax = fig.add_subplot(111) #format du graphique
ax.fill(X_plot[:,0], true_dens, fc='black', alpha=0.2, label='distribution en entrée')
ax.legend(loc='upper left')

#afficher la distribution des points en bas du graphique : afficher que pour N=10 ou N=100 sinon trop chargé
ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), "+k")

#limites du graphique : 
ax.set_xlim(-4, 9)
ax.set_ylim(-0.02, 0.3)

#Ajout de la légende : 
ax.set_xlabel("x")
ax.set_ylabel("Densité normalisée")

#sauvergarde du fichier : 
plt.savefig('2 pics (écart 2 sur le pic 2), distribution entrée.png', dpi=500)