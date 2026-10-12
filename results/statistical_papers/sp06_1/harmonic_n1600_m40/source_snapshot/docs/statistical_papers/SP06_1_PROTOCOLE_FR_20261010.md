# SP06.1 : calibration avec réponse, protocole fixé avant tirage

Cette expérience est distincte des pilotes SP05 et des études v5/v6. Elle teste une conséquence pratique de la variance D/n+BVB^T/m, sans revendiquer une nouvelle théorie générale d'inférence en deux étapes. Aucune réponse des jeux évalués ne sert à ajuster la loi des covariables ou à choisir une méthode.

## Modèle et unité

Chaque répétition contient n sujets de sélection iid et m autres sujets pilotes iid indépendants. Le graphe connu est une étoile connectée de p=12 angles directionnels, de centre 0 et d'arêtes (0,j), j=1,...,11. Le champ du centre est 3 fois (cos(0,7),sin(0,7)). Les autres champs sont 0,2 fois (cos(0,7+2*pi*j/11),sin(0,7+2*pi*j/11)). Les matrices sont M_0j=R(0,17*j) diag(0,8;0,5) R(−0,11*j)^T. Champs non nuls à tous les sommets ; matrices générales, singulières distinctes.

Le simulateur tire la racine par rejet exact, de proposition von Mises du champ central. Son poids est le produit des I0 des normes des champs naturels des feuilles ; la borne utilise ||h_j||+||M_0j||op. Les feuilles sont ensuite conditionnellement indépendantes von Mises. La validité du rejet est prouvée et testée séparément. Ce ne sont pas des états consécutifs d'un MCMC.

Le pilote ajuste conjointement les 24 champs scalaires et 44 coordonnées matricielles, avec M_j0=M_0j^T, par la pseudo-vraisemblance jointe du projet. Aucun graphe n'est appris. Le sandwich H^−1 K H^−1 utilise les scores joints par sujet et le Hessien analytique, y compris toutes les covariances entre nœuds. La perte moyenne est pénalisée par 0,5*||theta||²/m : le coefficient de pénalité est 1/m, donc son biais est négligeable à l'échelle racine m à dimension fixée. Le même réglage vaut pour tous les scénarios. Ce choix stabilise les petits pilotes sans constituer une garantie générale en grande dimension.

## Scénarios et statistiques

Cinq configurations, chacune avec 1 000 répétitions indépendantes, sont fixées :

1. n=m=400, réponse continue de second harmonique ;
2. n=1600, m=400, même réponse ;
3. n=m=400, réponse binaire non linéaire ;
4. n=m=400, nul complet avec réponse normale indépendante ;
5. n=1600, m=40, second harmonique, stress explicite du petit pilote.

Pour les trois configurations continues avec signal, l'ensemble vrai est {0,1,2,3,4,5}. La moyenne est 3/sqrt(6) fois la somme des cos(2*(X_j−0,7)) sur cet ensemble ; le bruit normal indépendant a écart type 1. Pour la réponse binaire, la probabilité vaut expit(1,5/sqrt(6) fois cette somme, sans le facteur 3, moins 0,25). Le nul complet a moyenne zéro, variance 1 et ensemble actif vide. Le premier harmonique seulement est utilisé pour la statistique de réponse, avec pénalité ridge fixée à 1. Un angle correspond à une seule hypothèse.

Cinq méthodes à q=0,1 : Wald avec variance pilote et BY ; même Wald sans variance pilote et BY ; Wald oracle et BY ; knockoff+ sur la loi ajustée ; knockoff+ oracle. Les deux méthodes Wald ajustées et knockoff+ ajusté partagent exactement le même témoin. Les méthodes oracle partagent un autre témoin, tiré de la vraie loi. Tous partagent originaux et réponse, permettant les contrastes appariés. La pénalité de régression est identique. Les conservatismes de BY et knockoff+ doivent être nommés ; les comparer au même niveau nominal ne les rend pas identiques.

## Sorties, erreurs et critère

Conserver à chaque répétition : graine par rôle, paramètres pilotes, sandwich complet, p-valeurs par angle, scores knockoff, sélections, FDP, puissance inconditionnelle, indicateur vide, fraction de tests nuls rejetés à 5 %, paramètres d'optimisation et erreurs explicites. Séparer temps de simulation, estimation, sandwich, génération et réponse ; conserver le pic mémoire. Les rejets du simulateur sont comptés. Aucun ajustement invalide n'est supprimé. Pour une méthode impossible à calculer, la procédure s'abstient à cette répétition et la sélection est vide ; ce comportement est compté dans les métriques inconditionnelles et le taux d'échec. Le rapport doit aussi donner des bornes sans hypothèse sur les contributions des échecs et ne doit pas créditer cette abstention comme une calibration réussie.

Contrôle central fixé : pour chaque configuration 1–4, la FDR moyenne de Wald corrigé ne doit pas dépasser 0,1 plus trois erreurs Monte-Carlo, et sa fréquence moyenne de rejets marginaux nuls à 5 % ne doit pas dépasser 0,05 plus trois erreurs Monte-Carlo. Ce diagnostic unilatéral permet le conservatisme ; il ne prouve pas le contrôle fini ni l'utilité. Le scénario 5 est une sensibilité conservée, sans exigence de réussite. Les échecs éventuels rendent le diagnostic de calibration incomplet. Les différences appariées, le coût de puissance, les résultats oracle et les sélections vides sont rapportés même si le critère est satisfait. Une réussite n'établit ni nouveauté, ni contrôle sous mauvaise spécification.

Les flux sont indépendants par configuration, répétition et rôle. Les écritures sont atomiques, les fichiers individuels numérotés et reprenables seulement avec configuration, code et environnement identiques. Après exécution : réagréger, rejouer des graines fixées (0,499,999), comparer toutes les valeurs numériques hormis les temps et conserver un registre SHA-256. Les éventuels changements de réglage exigent SP06.2, sans remplacer SP06.1.
