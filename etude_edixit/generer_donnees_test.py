# -*- coding: utf-8 -*-
"""
Générateur de données SIMULÉES pour l'étude eDixit.

Produit deux fichiers CSV avec exactement la structure décrite :
  - data/beneficiaires_edixit.csv   : clients à qui eDixit a été proposé en RDV (2025)
  - data/temoins_non_proposes.csv   : clients avec RDV en 2025 mais sans proposition eDixit

Structure : une ligne par client, colonne MOIS_RDV, variables d'évolution avec
suffixes _M6, _M3, _M1, _M, _P1, _P3, _P6.

Les données sont observées jusqu'à fin 2025 : les horizons post-RDV non encore
observables (ex. P6 pour un RDV de septembre 2025) sont à vide (NaN), comme dans
la réalité.

Un effet de l'outil est volontairement injecté chez les bénéficiaires
(épargne +, transfert depuis le compte à vue, équipement +, usage digital +)
afin que le pipeline d'étude ait quelque chose à détecter.
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

N_TRAITES = 4_000
N_TEMOINS = 12_000

SUFFIXES = ["_M6", "_M3", "_M1", "_M", "_P1", "_P3", "_P6"]
OFFSETS = {"_M6": -6, "_M3": -3, "_M1": -1, "_M": 0, "_P1": 1, "_P3": 3, "_P6": 6}

MOIS_2025 = [f"2025-{m:02d}" for m in range(1, 13)]
# Les RDV sont plus nombreux hors été
POIDS_MOIS = np.array([9, 10, 11, 10, 9, 8, 5, 4, 9, 10, 9, 6], dtype=float)
POIDS_MOIS /= POIDS_MOIS.sum()

CSP = ["Cadre", "Profession intermédiaire", "Employé", "Ouvrier", "Retraité", "Sans activité"]
SEGMENTS_MARCHE = ["Grand Public", "Intermédiaire", "Patrimonial"]
SEGMENTS_COMPORTEMENT = ["Digital", "Mixte", "Agence"]
REGIONS = ["Île-de-France", "Nord-Ouest", "Nord-Est", "Sud-Ouest", "Sud-Est", "Centre"]


def base_clients(n: int, traite: bool, prefixe_id: str) -> pd.DataFrame:
    """Variables statiques (socio-démo, segmentation) au moment du RDV.

    Un biais de sélection est injecté : les bénéficiaires d'eDixit sont un peu
    plus jeunes, plus digitaux et plus aisés que les témoins — c'est ce que le
    matching devra corriger.
    """
    df = pd.DataFrame({"ID_CLIENT": [f"{prefixe_id}{i:06d}" for i in range(1, n + 1)]})
    df["MOIS_RDV"] = rng.choice(MOIS_2025, size=n, p=POIDS_MOIS)

    if traite:
        df["AGE"] = np.clip(rng.normal(45, 12, n), 18, 90).round().astype(int)
        df["CSP"] = rng.choice(CSP, size=n, p=[0.24, 0.20, 0.24, 0.10, 0.16, 0.06])
        df["SEGMENT_MARCHE"] = rng.choice(SEGMENTS_MARCHE, size=n, p=[0.55, 0.30, 0.15])
        df["SEGMENT_COMPORTEMENTAL"] = rng.choice(SEGMENTS_COMPORTEMENT, size=n, p=[0.45, 0.35, 0.20])
    else:
        df["AGE"] = np.clip(rng.normal(50, 15, n), 18, 90).round().astype(int)
        df["CSP"] = rng.choice(CSP, size=n, p=[0.16, 0.18, 0.26, 0.13, 0.21, 0.06])
        df["SEGMENT_MARCHE"] = rng.choice(SEGMENTS_MARCHE, size=n, p=[0.65, 0.25, 0.10])
        df["SEGMENT_COMPORTEMENTAL"] = rng.choice(SEGMENTS_COMPORTEMENT, size=n, p=[0.30, 0.38, 0.32])

    df["SEXE"] = rng.choice(["F", "M"], size=n)
    df["ANCIENNETE_ANNEES"] = np.clip(rng.gamma(3.5, 4.0, n), 0, 60).round(1)
    df["REGION"] = rng.choice(REGIONS, size=n)
    return df


def trajectoires(df: pd.DataFrame, traite: bool) -> pd.DataFrame:
    """Variables bancaires en évolution autour du RDV (suffixes _M6 → _P6)."""
    n = len(df)
    aisance = {"Grand Public": 1.0, "Intermédiaire": 1.8, "Patrimonial": 4.0}
    fac = df["SEGMENT_MARCHE"].map(aisance).to_numpy() * (0.6 + df["AGE"].to_numpy() / 80)

    niveaux = {
        "ENCOURS_DAV": rng.lognormal(7.6, 0.7, n) * fac,          # ~2–10 k€
        "ENCOURS_EPARGNE": rng.lognormal(8.8, 1.0, n) * fac,      # ~5–40 k€
        "ENCOURS_CREDIT": rng.lognormal(8.0, 1.6, n) * (rng.random(n) < 0.55),
        "FLUX_ENTRANTS": rng.lognormal(7.7, 0.5, n) * (0.8 + 0.4 * fac / 4),
        "NB_PRODUITS": (rng.poisson(2.4, n) + 1).astype(float),
        "NB_CONNEXIONS_DIGITALES": rng.poisson(
            np.where(df["SEGMENT_COMPORTEMENTAL"] == "Digital", 22,
                     np.where(df["SEGMENT_COMPORTEMENTAL"] == "Mixte", 10, 3))
        ).astype(float),
    }

    # Effets injectés chez les bénéficiaires, croissants avec l'horizon (en %)
    effets = {
        "ENCOURS_EPARGNE": {1: 0.020, 3: 0.050, 6: 0.085},
        "ENCOURS_DAV": {1: -0.012, 3: -0.025, 6: -0.035},
        "ENCOURS_CREDIT": {1: 0.0, 3: 0.0, 6: 0.0},
        "FLUX_ENTRANTS": {1: 0.002, 3: 0.006, 6: 0.008},
        "NB_CONNEXIONS_DIGITALES": {1: 0.15, 3: 0.18, 6: 0.20},
        "NB_PRODUITS": {},  # géré à part (incréments discrets)
    }

    tendance = rng.normal(0.003, 0.004, n)  # dérive mensuelle commune ± bruit client

    out = {}
    for var, base in niveaux.items():
        for suf, off in OFFSETS.items():
            drift = (1 + tendance) ** off
            bruit = rng.normal(1, 0.05, n)
            val = base * drift * bruit
            if traite and off > 0:
                if var == "NB_PRODUITS":
                    p_equip = {1: 0.06, 3: 0.13, 6: 0.20}[off]
                    val = val + (rng.random(n) < p_equip)
                else:
                    val = val * (1 + effets[var].get(off, 0.0) + rng.normal(0, 0.01, n))
            if var.startswith("NB_"):
                val = np.round(np.maximum(val, 0))
            else:
                val = np.round(np.maximum(val, 0), 2)
            out[f"{var}{suf}"] = val

    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)


def censurer_horizons(df: pd.DataFrame) -> pd.DataFrame:
    """Met à NaN les horizons post-RDV non observables (données arrêtées à 2025-12)."""
    mois_num = df["MOIS_RDV"].str[5:7].astype(int)
    for suf, off in OFFSETS.items():
        if off > 0:
            non_obs = mois_num + off > 12
            cols = [c for c in df.columns if c.endswith(suf) and c[: -len(suf)] + "_M" in df.columns]
            df.loc[non_obs, cols] = np.nan
    return df


def saupoudrer_manquants(df: pd.DataFrame, taux: float = 0.012) -> pd.DataFrame:
    """~1 % de valeurs manquantes aléatoires sur les variables d'évolution."""
    cols = [c for c in df.columns if any(c.endswith(s) for s in SUFFIXES)]
    for c in rng.choice(cols, size=len(cols) // 3, replace=False):
        masque = rng.random(len(df)) < taux
        df.loc[masque, c] = np.nan
    return df


def main():
    traites = trajectoires(base_clients(N_TRAITES, True, "B"), traite=True)
    temoins = trajectoires(base_clients(N_TEMOINS, False, "T"), traite=False)

    # Anomalies volontaires pour tester les contrôles qualité :
    # 60 clients présents dans les deux tables + 25 doublons chez les témoins
    ids_communs = traites["ID_CLIENT"].sample(60, random_state=1).to_numpy()
    temoins.loc[temoins.index[:60], "ID_CLIENT"] = ids_communs
    temoins = pd.concat([temoins, temoins.sample(25, random_state=2)], ignore_index=True)

    traites = saupoudrer_manquants(censurer_horizons(traites))
    temoins = saupoudrer_manquants(censurer_horizons(temoins))

    traites.to_csv(DATA_DIR / "beneficiaires_edixit.csv", index=False, encoding="utf-8-sig")
    temoins.to_csv(DATA_DIR / "temoins_non_proposes.csv", index=False, encoding="utf-8-sig")
    print(f"OK - {len(traites)} bénéficiaires, {len(temoins)} témoins écrits dans {DATA_DIR}")


if __name__ == "__main__":
    main()
