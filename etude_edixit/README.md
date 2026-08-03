# Étude d'impact eDixit — RDV 2025

Pipeline Python qui mesure l'effet de l'outil eDixit en comparant les
bénéficiaires aux clients ayant eu un RDV sans proposition de l'outil, et
produit un classeur Excel mis en forme avec graphiques.

## Fichiers

| Fichier | Rôle |
|---|---|
| `generer_donnees_test.py` | Génère des données **simulées** de démonstration dans `data/` (à supprimer quand vous branchez les vraies données) |
| `etude_edixit.py` | Pipeline complet : qualité → descriptif → cohortes → matching → différence de différences → export Excel |
| `output/Etude_eDixit_2025.xlsx` | Résultat final (7 feuilles) |
| `output/figures/*.png` | Graphiques insérés dans le classeur |

## Brancher vos vraies données

1. Exportez vos deux tables en CSV (UTF-8) : une ligne par client.
2. Dans `etude_edixit.py`, section **CONFIGURATION**, adaptez :
   - `FICHIER_BENEFICIAIRES` et `FICHIER_TEMOINS` (chemins de vos fichiers) ;
   - `COL_ID` et `COL_MOIS` si vos colonnes ne s'appellent pas `ID_CLIENT` / `MOIS_RDV`
     (le mois de RDV doit être au format `AAAA-MM`).
3. Lancez `python etude_edixit.py`.

Les variables d'évolution sont **détectées automatiquement** : toute famille de
colonnes `X_M6, X_M3, X_M1, X_M, X_P1, X_P3, X_P6` présente dans les deux tables
est analysée (il faut au minimum `X_M` et un horizon post-RDV). Les colonnes
texte (segmentation, CSP, région…) sont automatiquement intégrées au score de
propension ; les horizons non observables doivent être laissés vides (NaN).

## Méthodologie

1. **Qualité** : dédoublonnage des identifiants, exclusion du groupe témoin des
   clients présents dans les deux tables, taux de valeurs manquantes.
2. **Descriptif** : profil des deux populations avant appariement (l'écart
   mesure le biais de sélection).
3. **Cohortes** : par mois de RDV, trajectoires M-6 → M+6 et indice base 100 au
   mois du RDV ; les horizons non encore observables apparaissent en « n.d. ».
4. **Appariement** : score de propension (régression logistique sur le profil
   socio-démo, la segmentation et les valeurs pré-RDV M-6 → M), appariement 1:1
   sans remise, **exact sur le mois de RDV**, caliper 0,2 écart-type du logit.
   Contrôle : différences moyennes standardisées (SMD) avant/après — objectif
   SMD < 0,10 après appariement.
5. **Impact** : différence de différences sur paires appariées —
   `effet = (X_bénéficiaire à M+h − X à M) − (X_témoin à M+h − X à M)` —
   avec IC à 95 % et test t. L'effet relatif est exprimé en % du niveau des
   bénéficiaires au mois du RDV.

## Limites à garder en tête

- Le matching ne corrige que les différences **observées** ; un biais peut
  subsister sur des caractéristiques non mesurées (ex. motivation du client).
- Les effets à M+6 ne sont mesurables que pour les cohortes de janvier à juin
  (données arrêtées à décembre 2025) : re-lancer l'étude quand les mois 2026
  seront disponibles pour consolider.
- Les encours à queue de distribution épaisse (crédit) donnent des IC larges ;
  une variante sur médianes ou en winsorisant est possible si besoin.

## Prérequis

```
pip install pandas numpy matplotlib scipy scikit-learn xlsxwriter
```
