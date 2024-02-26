import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

# ----------------------------------------------------------------------
# Exemple en 1D en additionant les résultats précécents : 
#nombre de points
N = 200
#paramètre de lissage
h=0.5
np.random.seed(1)
X = np.concatenate(
    (np.random.normal(0, 1, int(0.4 * N)), np.random.normal(5, 2, int(0.6 * N))))[:, np.newaxis]

X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
bins = np.linspace(-5, 10, 10)

true_dens = 0.4 * norm(0, 1).pdf(X_plot[:, 0]) + 0.6 * norm(5, 2).pdf(X_plot[:, 0])

fig, ax = plt.subplots()

# Tracé de la corube de données : 
ax.fill(X_plot[:, 0], true_dens, fc="black", alpha=0.2, label="distribution en entrée")
                                #regler comment est sombre le noir
# Tracé des KDE : 
colors = ["blue", "red","green","maroon","orange","purple"]
kernels = ["gaussian", "linear","epanechnikov","cosine","tophat","exponential"]
lw = 2

for color, kernel in zip(colors, kernels):
    kde = KernelDensity(kernel=kernel, bandwidth=h).fit(X)
    log_dens = kde.score_samples(X_plot)
    ax.plot(X_plot[:, 0],np.exp(log_dens),color=color,lw=lw,linestyle="-",label="kernel = '{0}'".format(kernel))

# Affichage du nombre de points du tracé : 
ax.text(6, 0.28, "N={0} points".format(N))
ax.text(6, 0.25, "h="+str(h))

# Positionnement de la légende pour le nom et la couleur des courbes tracées : 
ax.legend(loc="upper left")

#afficher la distribution des points en bas du graphique : afficher que pour N=10 ou N=100 sinon trop chargé
ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), "+k")

# Ajout de la légende : 
ax.set_xlabel("x")
ax.set_ylabel("Densité normalisée")

# Limites du graphique : 
ax.set_xlim(-4, 9)
ax.set_ylim(-0.02, 0.3)

# Sauvegarde de l'image créée : 
plt.savefig('superposition de plusieurs noyaux.png', dpi=500)