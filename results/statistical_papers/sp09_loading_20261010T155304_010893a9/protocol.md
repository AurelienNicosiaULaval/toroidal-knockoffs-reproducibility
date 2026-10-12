# SP09 : ordre des chargements et information latente

Protocole fixé avant le calcul numérique, 10 octobre 2026. Cette étape complète l'évaluation scientifique SP08; elle ne relance pas les expériences historiques et ne choisit aucun réglage avec des réponses évaluées.

## Question et décision

À loi bilinéaire connue et fixée, comparer deux noyaux de redessinage gaussien A=J+D, avec D_jj=d_j I_2 et A positive semi-définie. Chercher un ordre pour toutes les fonctions f(X), plutôt qu'une baisse de corrélation marginale présentée comme un gain de puissance.

La conjecture à examiner est que l'ordre de covariance à un pas des noyaux est exactement l'ordre coordonnée par coordonnée des chargements d. Une augmentation de tous les chargements devrait augmenter E[f(X)f(témoin)] et diminuer la variance conditionnelle après observation du latent. Les allocations par poids de sommets non proportionnels pourraient alors être incomparables pour cet ordre, même si l'une minimise la charge maximale.

Ce critère concerne la covariance du noyau et la prédiction à partir du latent. Il n'est ni la reconstructibilité de X à partir des autres originaux et des témoins, ni un ordre de puissance, ni une garantie de FDR avec loi estimée. Les arguments d'ordre d'information et d'opérateurs ont des antécédents classiques; leur spécialisation précise reste une contribution candidate, sans certificat de priorité.

Poursuivre vers une note et éventuellement un court passage du manuscrit si la preuve, le traitement des matrices singulières et les calculs indépendants concordent. Restreindre ou retirer l'énoncé si une étape échoue. Conserver tout résultat défavorable. Aucune simulation de sélection n'est prévue.

## Vérifications fixées

1. Prouver le pont gaussien G_1|G_2 avec pseudo-inverse, et la chaîne X -> G_2 -> G_1 lorsque A_2-A_1 est positive semi-définie.
2. Déduire l'ordre de toutes les covariances à un pas et du résidu d'information, éventuellement conditionné par X_-N. Vérifier que cela ne donne pas un ordre des variances pilotes réelles BVB^T.
3. Examiner la nécessité de l'ordre des d_j à l'aide d'une fonction cos(X_j) et de la suffisance du seul champ F_j pour X_j. Examiner les allocations s_jk u_k/u_j sans répéter les anciennes expériences de Perron.
4. Cas univarié uniforme : quadrature indépendante de la densité radiale de Rice pour les corrélations des harmoniques 1, 2 et 3, chargements 0,125; 0,5; 1; 4; 16. Contrôler l'intégrale de densité et l'erreur de quadrature. Comparer avec 100000 couples iid par chargement, sans MCMC.
5. Étoile continue à trois sommets, champs non nuls et deux matrices anisotropes de norme 0,55 : 100000 originaux iid obtenus par rejet uniforme avec enveloppe analytique. Construire les champs sans utiliser les générateurs du projet. Comparer A_1 et A_2=A_1+0,4I, y compris A_1 singulière, et vérifier les identités de covariance des moyennes conditionnelles sur les statistiques canoniques et quelques harmoniques. Conserver les erreurs Monte-Carlo; tolérance diagnostique de cinq erreurs types, sans la présenter comme preuve ou certificat simultané.
6. Contrôle non comparable : allocation symétrique contre allocation équilibrée de cette étoile. Vérifier que les innovations du centre et des feuilles évoluent dans des sens opposés. Les originaux sont les mêmes et les différences sont appariées.

Graines indépendantes et fixées : SeedSequence(2026101009), enfants distincts pour cas univariés, tirage des originaux, bruit des champs et témoins. Sauvegarder configuration, code et empreintes, sorties de quadrature, métriques et erreurs Monte-Carlo, nombre de propositions et d'acceptations, durées et échecs. Le programme ne doit pas importer les simulateurs ou les générateurs existants. Cette étude est un contrôle de noyaux connus, pas une expérience de FDR.

## Antériorité à lire

Hobert et Marchev (2008), Annals of Statistics 36(2), 532-554, DOI 10.1214/009053607000000569, §§2-3, identités d'espérance conditionnelle et ordre de covariance. Blackwell (1953), Annals of Mathematical Statistics 24(2), 265-272, DOI 10.1214/aoms/1177729032, si texte accessible. Distinguer la comparaison générale d'expériences et la spécialisation aux chargements gaussiens toroïdaux. Si seul le résumé est accessible, le signaler.
