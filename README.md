# ST2 (Théorie des Jeux) - EI Algorithmique Génétique

## Description  
Ce projet implémente deux algorithmes d'optimisation :

    - **Recuit simulé**  
    - **Algorithme génétique**  
Le but est d'ajuster les rotations de nucléotides afin de rendre une séquence d’ADN circulaire.

---

## Installation  
Installez les dépendances requises (**numpy** et **matplotlib**) avec :  

```bash
pip install -r requirements.txt
```

## Utilisation

**Lancez le programme avec :**

```
python -m 3dna data/plasmid_8k.fasta --genetic --annealing --generations 200 --population 100 --mutation_rate 0.1
```

**Options disponibles**

    --genetic : Active l’algorithme génétique.
    --annealing : Active le recuit simulé.
    --generations <int> : Nombre de générations (défaut: 100).
    --population <int> : Taille de la population (défaut: 50).
    --mutation_rate <float> : Probabilité de mutation par gène (ex: 0.1).

**Affichez l’aide avec :**

```
python -m 3dna --help
```

**Exemple de résultat**

Après exécution, voici un exemple de graphe obtenu :
![Exemple de graphe](data/example.png)
