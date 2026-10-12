# Contrôle borné de la preuve SP13

Protocole fixé avant exécution, 10 octobre 2026. Question: le remplacement de l'alignement de tous les signes par une proportion tendant vers un justifie-t-il le régime suffisant n/(K²m) tendant vers l'infini?

La preuve doit fournir séparément l'erreur globale de régression, l'erreur quadratique moyenne des scores nuls, le calcul du vrai seuil, une borne valable pour tout couplage à une référence de signes exacts, et l'échelle de fluctuation de l'oracle. La note SP12 et ses neuf diagnostics restent inchangés.

Le programme de contrôle utilise des vecteurs de scores déterministes. Il énumère les lois de signes de références, y compris des zéros, avec K = 4, 8, 12 et 16. Les probabilités sont des comptes exacts, sans erreur Monte-Carlo. Une référence à signes communs fournit un contrôle négatif: elle doit violer le certificat de signes indépendants. Des scores avec K = 20, 100 et 1000, des alternatives fortes et une proportion prédéfinie de signes perturbés vérifient le seuil knockoff+ et les bornes de FDP. Un seul signe perturbé montre que l'erreur quadratique moyenne peut s'annuler sans alignement de tous les signes.

Le calcul doit conserver configurations, scores déterministes, comptes, empreintes, durée et reçu. Il n'est pas une simulation de covariables toroïdales, ne mesure pas une FDR de population et ne certifie pas la limite asymptotique par des chiffres.

Critère de réussite: toutes les bornes et identités déterministes concordent, le contrôle négatif échoue là où attendu, et la preuve explicite la non-singularité et les facteurs K dans les restes. Critère de révision: une incohérence de seuil, de compte ou de normalisation reste consignée; aucune intégration comme résultat démontré avant correction.

Une lecture complémentaire d'ARK vise ses Conditions 1 et 5, équation (7), dans la version du 4 juin 2024. La comparaison doit montrer que le seuil choisi est dans son intervalle pertinent. Une obstruction pour un oracle spécifié ne sera pas annoncée comme une contradiction du théorème d'ARK.
