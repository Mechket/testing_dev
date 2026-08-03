# -*- coding: utf-8 -*-
"""
Étude de l'effet de l'outil eDixit — pipeline complet.

Entrées  : deux tables « une ligne par client » (bénéficiaires / témoins), avec
           une colonne mois de RDV et des variables d'évolution suffixées
           _M6, _M3, _M1, _M, _P1, _P3, _P6.
Sorties  : output/Etude_eDixit_2025.xlsx (tables + graphiques)
           output/figures/*.png

Étapes :
  1. Chargement et contrôles qualité (doublons, clients dans les deux tables,
     valeurs manquantes, horizons non observables)
  2. Descriptif des deux populations
  3. Analyse de cohortes (par mois de RDV, trajectoires M-6 → M+6)
  4. Appariement bénéficiaires/témoins par score de propension
     (exact sur le mois de RDV + plus proche voisin sous caliper)
  5. Impact causal par différence de différences sur l'échantillon apparié
  6. Export Excel mis en forme avec graphiques

Pour brancher vos vraies données : modifiez FICHIER_BENEFICIAIRES et
FICHIER_TEMOINS ci-dessous (et COL_ID / COL_MOIS si vos noms diffèrent).
Les variables d'évolution sont détectées automatiquement via leurs suffixes.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import xlsxwriter
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# CONFIGURATION — à adapter à vos fichiers réels
# ---------------------------------------------------------------------------
RACINE = Path(__file__).parent
FICHIER_BENEFICIAIRES = RACINE / "data" / "beneficiaires_edixit.csv"
FICHIER_TEMOINS = RACINE / "data" / "temoins_non_proposes.csv"

COL_ID = "ID_CLIENT"
COL_MOIS = "MOIS_RDV"          # mois du RDV, format 'AAAA-MM'

# Suffixes temporels (du plus long au plus court pour la détection)
SUFFIXES = ["_M6", "_M3", "_M1", "_P1", "_P3", "_P6", "_M"]
OFFSETS = {"_M6": -6, "_M3": -3, "_M1": -1, "_M": 0, "_P1": 1, "_P3": 3, "_P6": 6}
SUFFIXES_ORDRE = ["_M6", "_M3", "_M1", "_M", "_P1", "_P3", "_P6"]
HORIZONS_POST = ["_P1", "_P3", "_P6"]
CALIPER = 0.2                   # en écarts-types du logit du score de propension

SORTIE = RACINE / "output"
DOSSIER_FIG = SORTIE / "figures"
FICHIER_EXCEL = SORTIE / "Etude_eDixit_2025.xlsx"

# ---------------------------------------------------------------------------
# Style graphique (palette validée, mode clair)
# ---------------------------------------------------------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLEU, ORANGE = "#2a78d6", "#eb6834"
RAMPE_BLEUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
               "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.linewidth": 1.0,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.titlesize": 11, "axes.titlelocation": "left", "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "legend.frameon": False, "legend.fontsize": 9,
})


def fmt_nombre(x, dec=0):
    if pd.isna(x):
        return "n.d."
    return f"{x:,.{dec}f}".replace(",", " ").replace(".", ",")


def formater_axe_y(ax):
    """Ticks avec un nombre de décimales adapté à l'étendue de l'axe
    (évite les libellés dupliqués « 3, 3, 4, 4 » sur les petites échelles)."""
    def _fmt(v, _):
        etendue = ax.get_ylim()[1] - ax.get_ylim()[0]
        dec = 0 if etendue >= 20 else (1 if etendue >= 2 else 2)
        return fmt_nombre(v, dec)
    ax.yaxis.set_major_formatter(_fmt)


# ---------------------------------------------------------------------------
# 1. Chargement + contrôles qualité
# ---------------------------------------------------------------------------
def detecter_stems(df: pd.DataFrame):
    """Détecte les variables d'évolution : un « stem » est retenu si la colonne
    stem_M existe et au moins un horizon post-RDV."""
    stems = {}
    for col in df.columns:
        for suf in SUFFIXES:
            if col.endswith(suf):
                stems.setdefault(col[: -len(suf)], set()).add(suf)
                break
    return sorted(s for s, sufs in stems.items()
                  if "_M" in sufs and sufs & set(HORIZONS_POST))


def charger():
    traites = pd.read_csv(FICHIER_BENEFICIAIRES, encoding="utf-8-sig")
    temoins = pd.read_csv(FICHIER_TEMOINS, encoding="utf-8-sig")
    qualite = []

    for nom, df in [("Bénéficiaires", traites), ("Témoins", temoins)]:
        qualite.append((f"{nom} : lignes chargées", len(df), ""))

    # Doublons d'identifiant
    for nom, df in [("Bénéficiaires", traites), ("Témoins", temoins)]:
        nb_dbl = df.duplicated(COL_ID).sum()
        qualite.append((f"{nom} : doublons {COL_ID}", nb_dbl,
                        "supprimés (1re occurrence conservée)" if nb_dbl else "OK"))
    traites = traites.drop_duplicates(COL_ID)
    temoins = temoins.drop_duplicates(COL_ID)

    # Clients présents dans les deux tables -> exclus des témoins
    communs = set(traites[COL_ID]) & set(temoins[COL_ID])
    qualite.append(("Clients présents dans les 2 tables", len(communs),
                    "exclus du groupe témoin" if communs else "OK"))
    temoins = temoins[~temoins[COL_ID].isin(communs)].copy()

    stems = detecter_stems(traites)
    stems_t = detecter_stems(temoins)
    if set(stems) != set(stems_t):
        qualite.append(("Variables d'évolution non communes",
                        len(set(stems) ^ set(stems_t)), "restreint à l'intersection"))
        stems = sorted(set(stems) & set(stems_t))
    qualite.append(("Variables d'évolution détectées", len(stems), ", ".join(stems)))

    # Taux de valeurs manquantes maximum (hors horizons non observables)
    cols_evo = [f"{s}{suf}" for s in stems for suf in ["_M6", "_M3", "_M1", "_M"]]
    for nom, df in [("Bénéficiaires", traites), ("Témoins", temoins)]:
        pct = df[cols_evo].isna().mean().max() * 100
        qualite.append((f"{nom} : % manquants max (pré-RDV)", round(pct, 2),
                        "OK" if pct < 5 else "à vérifier"))

    traites["TRAITE"] = 1
    temoins["TRAITE"] = 0
    return traites, temoins, stems, qualite


# ---------------------------------------------------------------------------
# 2. Descriptif
# ---------------------------------------------------------------------------
def descriptif(traites, temoins, stems):
    lignes = []

    def pct(df, col, val):
        return (df[col] == val).mean() * 100 if col in df else np.nan

    paires = [("Effectif", lambda d: len(d), 0),
              ("Âge moyen", lambda d: d["AGE"].mean() if "AGE" in d else np.nan, 1),
              ("% Femmes", lambda d: pct(d, "SEXE", "F"), 1),
              ("Ancienneté moyenne (années)",
               lambda d: d["ANCIENNETE_ANNEES"].mean() if "ANCIENNETE_ANNEES" in d else np.nan, 1),
              ("% segment Patrimonial", lambda d: pct(d, "SEGMENT_MARCHE", "Patrimonial"), 1),
              ("% profil Digital", lambda d: pct(d, "SEGMENT_COMPORTEMENTAL", "Digital"), 1)]
    for stem in stems:
        paires.append((f"{stem} moyen à M", lambda d, s=stem: d[f"{s}_M"].mean(), 0))

    for libelle, f, dec in paires:
        vt, vc = f(traites), f(temoins)
        lignes.append({"Indicateur": libelle, "Bénéficiaires": round(vt, dec),
                       "Témoins": round(vc, dec),
                       "Écart": round(vt - vc, dec) if pd.notna(vt) and pd.notna(vc) else np.nan})
    profil = pd.DataFrame(lignes)

    effectifs = (pd.concat([traites.assign(Groupe="Bénéficiaires"),
                            temoins.assign(Groupe="Témoins")])
                 .groupby([COL_MOIS, "Groupe"]).size().unstack(fill_value=0)
                 .reindex(sorted(set(traites[COL_MOIS]) | set(temoins[COL_MOIS]))))
    return profil, effectifs


# ---------------------------------------------------------------------------
# 3. Cohortes
# ---------------------------------------------------------------------------
def analyse_cohortes(traites, stems):
    """Par cohorte de mois de RDV : moyenne de chaque variable à chaque point,
    et indice base 100 au mois du RDV (M)."""
    moyennes, indices = {}, {}
    for stem in stems:
        cols = {suf: f"{stem}{suf}" for suf in SUFFIXES_ORDRE if f"{stem}{suf}" in traites}
        m = traites.groupby(COL_MOIS)[list(cols.values())].mean()
        m.columns = [f"M{OFFSETS[s]:+d}".replace("+0", "") if OFFSETS[s] else "M"
                     for s in cols]
        moyennes[stem] = m
        base = m["M"]
        indices[stem] = m.div(base, axis=0) * 100
    return moyennes, indices


# ---------------------------------------------------------------------------
# 4. Appariement par score de propension
# ---------------------------------------------------------------------------
def preparer_features(df, stems, cat_cols, num_cols):
    X = pd.get_dummies(df[cat_cols].astype(str), drop_first=False) if cat_cols else pd.DataFrame(index=df.index)
    pre = [f"{s}{suf}" for s in stems for suf in ["_M6", "_M3", "_M1", "_M"] if f"{s}{suf}" in df]
    num = df[num_cols + pre].copy()
    num = num.fillna(num.median(numeric_only=True))
    return pd.concat([num, X.astype(float)], axis=1)


def apparier(traites, temoins, stems):
    exclues = {COL_ID, COL_MOIS, "TRAITE"}
    est_num = lambda c: pd.api.types.is_numeric_dtype(traites[c])
    cat_cols = [c for c in traites.columns
                if not est_num(c) and c not in exclues
                and traites[c].nunique() <= 30 and c in temoins.columns]
    num_cols = [c for c in traites.columns
                if c not in exclues and est_num(c)
                and not any(c.endswith(s) for s in SUFFIXES)
                and c in temoins.columns]

    tout = pd.concat([traites, temoins], ignore_index=True)
    X = preparer_features(tout, stems, cat_cols, num_cols)
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=tout.index)
    y = tout["TRAITE"].to_numpy()

    modele = LogisticRegression(max_iter=2000, C=1.0)
    modele.fit(X, y)
    ps = modele.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, ps)
    logit = np.log(np.clip(ps, 1e-6, 1 - 1e-6) / (1 - np.clip(ps, 1e-6, 1 - 1e-6)))
    tout = tout.assign(_PS=ps, _LOGIT=logit)
    caliper_abs = CALIPER * logit.std()

    # Appariement 1:1 sans remise, exact sur le mois de RDV, glouton sur le logit
    paires = []
    for mois, grp in tout.groupby(COL_MOIS):
        gt = grp[grp["TRAITE"] == 1]
        gc = grp[grp["TRAITE"] == 0]
        if gt.empty or gc.empty:
            continue
        arbre = cKDTree(gc["_LOGIT"].to_numpy().reshape(-1, 1))
        libres = np.ones(len(gc), dtype=bool)
        # les scores les plus élevés (plus durs à apparier) d'abord
        for i in gt.sort_values("_PS", ascending=False).index:
            k = min(len(gc), 80)
            dist, idx = arbre.query([[tout.at[i, "_LOGIT"]]], k=k)
            for d, j in zip(np.atleast_1d(dist.ravel()), np.atleast_1d(idx.ravel())):
                if d > caliper_abs:
                    break
                if libres[j]:
                    libres[j] = False
                    paires.append((i, gc.index[j]))
                    break

    idx_t = [p[0] for p in paires]
    idx_c = [p[1] for p in paires]
    infos = {"auc": auc, "caliper": caliper_abs, "n_traites": int((y == 1).sum()),
             "n_apparies": len(paires),
             "taux": len(paires) / max((y == 1).sum(), 1) * 100}

    # Équilibre avant/après (différences moyennes standardisées)
    smd = []
    m_t, m_c = y == 1, y == 0
    apres_t = X.loc[idx_t]
    apres_c = X.loc[idx_c]
    for col in X.columns:
        et = np.sqrt((X.loc[m_t, col].var() + X.loc[m_c, col].var()) / 2)
        if et == 0 or np.isnan(et):
            continue
        smd.append({"Variable": col,
                    "SMD avant": abs(X.loc[m_t, col].mean() - X.loc[m_c, col].mean()) / et,
                    "SMD après": abs(apres_t[col].mean() - apres_c[col].mean()) / et})
    smd = pd.DataFrame(smd).sort_values("SMD avant", ascending=False)

    apparie_t = tout.loc[idx_t].reset_index(drop=True)
    apparie_c = tout.loc[idx_c].reset_index(drop=True)
    return apparie_t, apparie_c, smd, infos


# ---------------------------------------------------------------------------
# 5. Différence de différences sur l'échantillon apparié
# ---------------------------------------------------------------------------
def impact_did(apparie_t, apparie_c, stems):
    """Pour chaque variable et chaque horizon : effet = (évolution bénéficiaire)
    - (évolution témoin apparié), entre M et M+h. IC à 95 % et test t apparié."""
    lignes = []
    for stem in stems:
        for suf in HORIZONS_POST:
            ct, cm = f"{stem}{suf}", f"{stem}_M"
            if ct not in apparie_t:
                continue
            d = ((apparie_t[ct] - apparie_t[cm]) - (apparie_c[ct] - apparie_c[cm])).dropna()
            if len(d) < 30:
                continue
            m, se = d.mean(), d.std() / np.sqrt(len(d))
            tcrit = stats.t.ppf(0.975, len(d) - 1)
            p = stats.ttest_1samp(d, 0).pvalue
            base = apparie_t.loc[d.index, cm].mean()
            lignes.append({"Variable": stem, "Horizon": f"M+{OFFSETS[suf]}",
                           "N paires": len(d), "Effet": m,
                           "IC95 bas": m - tcrit * se, "IC95 haut": m + tcrit * se,
                           "Effet relatif (%)": m / base * 100 if base else np.nan,
                           "p-value": p, "Significatif (5 %)": "Oui" if p < 0.05 else "Non"})
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
def unite(stem):
    return "€" if stem.startswith(("ENCOURS", "FLUX")) else ""


def fig_effectifs(effectifs):
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    x = np.arange(len(effectifs))
    largeur = 0.38
    ax.bar(x - largeur / 2, effectifs.get("Bénéficiaires", 0), largeur,
           color=BLEU, label="Bénéficiaires eDixit", zorder=3)
    ax.bar(x + largeur / 2, effectifs.get("Témoins", 0), largeur,
           color=ORANGE, label="Témoins (RDV sans proposition)", zorder=3)
    ax.set_xticks(x, [m[5:] + "/25" for m in effectifs.index])
    ax.grid(axis="x", visible=False)
    ax.set_title("Effectifs par mois de RDV (cohortes 2025)", pad=12)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(DOSSIER_FIG / "fig_effectifs.png", dpi=144)
    plt.close(fig)


def fig_trajectoires(apparie_t, apparie_c, stems):
    n = len(stems)
    ncols = 2
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.1 * nrows))
    axes = np.atleast_1d(axes).ravel()
    xs = [OFFSETS[s] for s in SUFFIXES_ORDRE]
    labels = ["M-6", "M-3", "M-1", "M", "M+1", "M+3", "M+6"]
    for ax, stem in zip(axes, stems):
        # Composition constante : uniquement les paires observables sur les 7 points,
        # sinon les moyennes mélangent des cohortes différentes selon l'horizon.
        cols = [f"{stem}{s}" for s in SUFFIXES_ORDRE if f"{stem}{s}" in apparie_t]
        complet = (apparie_t[cols].notna().all(axis=1)
                   & apparie_c[cols].notna().all(axis=1))
        for df, coul, nom in [(apparie_t, BLEU, "Bénéficiaires"),
                              (apparie_c, ORANGE, "Témoins appariés")]:
            ys = [df.loc[complet, f"{stem}{s}"].mean() if f"{stem}{s}" in df else np.nan
                  for s in SUFFIXES_ORDRE]
            ax.plot(xs, ys, color=coul, lw=2, marker="o", ms=5, label=nom, zorder=3)
        ax.axvline(0, color=BASELINE, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.set_xticks(xs, labels)
        u = unite(stem)
        ax.set_title(stem + (f" ({u})" if u else "")
                     + f" — {fmt_nombre(complet.sum())} paires", pad=8)
        formater_axe_y(ax)
    for ax in axes[n:]:
        ax.set_visible(False)
    manches, noms = axes[0].get_legend_handles_labels()
    fig.legend(manches, noms, loc="upper right", bbox_to_anchor=(0.99, 0.965), ncols=2)
    fig.suptitle("Trajectoires moyennes autour du RDV — paires appariées à historique\n"
                 "complet (cohortes observables jusqu'à M+6) ; pointillé = mois du RDV",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(DOSSIER_FIG / "fig_trajectoires.png", dpi=144)
    plt.close(fig)


def fig_heatmap_cohortes(indices, stem):
    tab = indices[stem]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    cmap = mcolors.LinearSegmentedColormap.from_list("bleu", RAMPE_BLEUE)
    cmap.set_bad("#f0efec")
    vals = tab.to_numpy(dtype=float)
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    im = ax.imshow(vals, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(tab.shape[1]), tab.columns)
    ax.set_yticks(range(tab.shape[0]), [m[5:] + "/25" for m in tab.index])
    ax.grid(visible=False)
    seuil = vmin + 0.55 * (vmax - vmin)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = vals[i, j]
            if np.isnan(v):
                ax.text(j, i, "n.d.", ha="center", va="center", fontsize=7.5, color=MUTED)
            else:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if v > seuil else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, color=MUTED, labelcolor=MUTED)
    ax.set_title(f"{stem} — indice base 100 au mois du RDV, par cohorte (bénéficiaires)",
                 pad=12)
    ax.set_xlabel("Position par rapport au RDV")
    fig.tight_layout()
    fig.savefig(DOSSIER_FIG / "fig_cohortes_heatmap.png", dpi=144)
    plt.close(fig)


def fig_did(did):
    stems_fig = did["Variable"].unique()
    n = len(stems_fig)
    ncols = 3
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 3.3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, stem in zip(axes, stems_fig):
        sub = did[did["Variable"] == stem].reset_index(drop=True)
        x = np.arange(len(sub))
        err = [sub["Effet"] - sub["IC95 bas"], sub["IC95 haut"] - sub["Effet"]]
        ax.bar(x, sub["Effet"], 0.52, color=BLEU, zorder=3)
        ax.errorbar(x, sub["Effet"], yerr=err, fmt="none", ecolor=INK2,
                    elinewidth=1.2, capsize=3, zorder=4)
        ax.axhline(0, color=BASELINE, lw=1, zorder=2)
        haut = max(sub["IC95 haut"].max(), 0)
        bas = min(sub["IC95 bas"].min(), 0)
        marge = (haut - bas) * 0.18 or 1
        for xi, (_, r) in zip(x, sub.iterrows()):
            etoile = " *" if r["p-value"] < 0.05 else ""
            if r["Effet"] >= 0:
                ax.text(xi, r["IC95 haut"] + marge * 0.15, fmt_nombre(r["Effet"], 1) + etoile,
                        ha="center", va="bottom", fontsize=8.5, color=INK)
            else:
                ax.text(xi, r["IC95 bas"] - marge * 0.15, fmt_nombre(r["Effet"], 1) + etoile,
                        ha="center", va="top", fontsize=8.5, color=INK)
        ax.set_ylim(bas - marge, haut + marge)
        ax.set_xticks(x, sub["Horizon"])
        ax.grid(axis="x", visible=False)
        u = unite(stem)
        ax.set_title(stem + (f" ({u})" if u else ""), pad=8)
        formater_axe_y(ax)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Effet net d'eDixit par horizon — différence de différences,"
                 " IC à 95 % (* = significatif à 5 %)",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(DOSSIER_FIG / "fig_did.png", dpi=144)
    plt.close(fig)


def fig_balance(smd, top=15):
    sub = smd.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 0.34 * len(sub) + 1.6))
    y = np.arange(len(sub))
    ax.scatter(sub["SMD avant"], y, color=ORANGE, s=45, label="Avant appariement", zorder=3)
    ax.scatter(sub["SMD après"], y, color=BLEU, s=45, label="Après appariement", zorder=4)
    for yi, (_, r) in zip(y, sub.iterrows()):
        ax.plot([r["SMD après"], r["SMD avant"]], [yi, yi], color=GRID, lw=1.4, zorder=2)
    ax.axvline(0.10, color=BASELINE, lw=1, ls=(0, (4, 3)))
    ax.text(0.10, len(sub) - 0.2, " seuil 0,10", fontsize=8, color=MUTED, va="bottom")
    ax.set_yticks(y, sub["Variable"], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Différence moyenne standardisée (valeur absolue)")
    ax.set_title("Équilibre bénéficiaires / témoins avant et après appariement", pad=12)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(DOSSIER_FIG / "fig_balance.png", dpi=144)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Export Excel
# ---------------------------------------------------------------------------
class ClasseurEtude:
    def __init__(self, chemin):
        self.wb = xlsxwriter.Workbook(str(chemin), {"nan_inf_to_errors": True})
        s = self.wb.add_format
        self.f_titre = s({"bold": True, "font_size": 16, "font_color": "#0b0b0b"})
        self.f_sous = s({"font_color": "#52514e", "font_size": 10, "text_wrap": True})
        self.f_h2 = s({"bold": True, "font_size": 12, "font_color": "#0b0b0b"})
        self.f_head = s({"bold": True, "font_color": "#ffffff", "bg_color": "#2a78d6",
                         "border": 1, "border_color": "#1c5cab", "text_wrap": True,
                         "valign": "vcenter"})
        self.f_cell = s({"border": 1, "border_color": "#e1e0d9"})
        self.f_txt = s({"border": 1, "border_color": "#e1e0d9", "text_wrap": True})
        self.f_int = s({"border": 1, "border_color": "#e1e0d9", "num_format": "#,##0"})
        self.f_dec = s({"border": 1, "border_color": "#e1e0d9", "num_format": "#,##0.0"})
        self.f_dec2 = s({"border": 1, "border_color": "#e1e0d9", "num_format": "0.00"})
        self.f_pval = s({"border": 1, "border_color": "#e1e0d9", "num_format": "0.0000"})
        self.f_kpi_val = s({"bold": True, "font_size": 20, "font_color": "#2a78d6"})
        self.f_kpi_lib = s({"font_color": "#52514e", "font_size": 9, "text_wrap": True,
                            "valign": "top"})
        self.f_sig = s({"border": 1, "border_color": "#e1e0d9", "font_color": "#006300",
                        "bold": True})
        self.f_nsig = s({"border": 1, "border_color": "#e1e0d9", "font_color": "#898781"})

    def feuille(self, nom, titre, sous_titre=""):
        ws = self.wb.add_worksheet(nom)
        ws.hide_gridlines(2)
        ws.set_column(0, 0, 2)
        ws.write(1, 1, titre, self.f_titre)
        if sous_titre:
            ws.merge_range(2, 1, 2, 10, sous_titre, self.f_sous)
        return ws

    def table(self, ws, df, lig, col, formats=None, largeurs=None):
        """Écrit un DataFrame avec en-tête stylé. formats : dict colonne->format."""
        formats = formats or {}
        for j, c in enumerate(df.columns):
            ws.write(lig, col + j, str(c), self.f_head)
            larg = (largeurs or {}).get(c, max(len(str(c)) + 2,
                    int(df[c].astype(str).str.len().quantile(0.9)) + 2 if len(df) else 10))
            ws.set_column(col + j, col + j, min(larg, 42))
        for i, (_, r) in enumerate(df.iterrows()):
            for j, c in enumerate(df.columns):
                v = r[c]
                fmt = formats.get(c, self.f_cell)
                if pd.isna(v):
                    ws.write_blank(lig + 1 + i, col + j, None, self.f_cell)
                elif isinstance(v, (int, np.integer)):
                    ws.write_number(lig + 1 + i, col + j, int(v),
                                    formats.get(c, self.f_int))
                elif isinstance(v, (float, np.floating)):
                    ws.write_number(lig + 1 + i, col + j, float(v), fmt)
                else:
                    ws.write(lig + 1 + i, col + j, str(v), fmt)
        return lig + 1 + len(df)

    def image(self, ws, lig, col, chemin, echelle=0.72):
        ws.insert_image(lig, col, str(chemin), {"x_scale": echelle, "y_scale": echelle})

    def kpi(self, ws, lig, col, valeur, libelle):
        ws.write(lig, col, valeur, self.f_kpi_val)
        ws.merge_range(lig + 1, col, lig + 2, col + 1, libelle, self.f_kpi_lib)


def exporter_excel(qualite, profil, effectifs, moyennes, indices, stem_heatmap,
                   smd, infos, did):
    cl = ClasseurEtude(FICHIER_EXCEL)

    # --- Synthèse -----------------------------------------------------------
    ws = cl.feuille("Synthèse", "Étude d'impact de l'outil eDixit — RDV 2025",
                    "Comparaison bénéficiaires vs témoins appariés (score de propension, "
                    "appariement exact sur le mois de RDV) ; effet mesuré en différence de "
                    "différences entre le mois du RDV (M) et M+1 / M+3 / M+6.")
    epargne6 = did[(did["Variable"] == stem_heatmap) & (did["Horizon"] == "M+6")]
    eff6 = epargne6["Effet"].iloc[0] if len(epargne6) else np.nan
    rel6 = epargne6["Effet relatif (%)"].iloc[0] if len(epargne6) else np.nan
    n_sig = int((did["p-value"] < 0.05).sum())
    cl.kpi(ws, 4, 1, fmt_nombre(infos["n_traites"]), "Bénéficiaires eDixit (2025)")
    cl.kpi(ws, 4, 3, fmt_nombre(infos["n_apparies"]),
           f"paires appariées ({infos['taux']:.0f} % des bénéficiaires)")
    cl.kpi(ws, 4, 5, f"{fmt_nombre(eff6)} €" if pd.notna(eff6) else "n.d.",
           f"effet net sur {stem_heatmap} à M+6 ({rel6:+.1f} %)" if pd.notna(rel6)
           else f"effet net sur {stem_heatmap} à M+6")
    cl.kpi(ws, 4, 7, f"{n_sig}/{len(did)}",
           "effets significatifs à 5 % (variables × horizons)")

    ws.write(9, 1, "Principaux effets estimés à M+6", cl.f_h2)
    did6 = did[did["Horizon"] == "M+6"][["Variable", "Effet", "IC95 bas", "IC95 haut",
                                          "Effet relatif (%)", "p-value",
                                          "Significatif (5 %)"]]
    fin = cl.table(ws, did6, 10, 1,
                   formats={"Effet": cl.f_dec, "IC95 bas": cl.f_dec, "IC95 haut": cl.f_dec,
                            "Effet relatif (%)": cl.f_dec2, "p-value": cl.f_pval})
    for i, (_, r) in enumerate(did6.iterrows()):
        fmt = cl.f_sig if r["Significatif (5 %)"] == "Oui" else cl.f_nsig
        ws.write(11 + i, 7, r["Significatif (5 %)"], fmt)
    ws.merge_range(fin + 2, 1, fin + 2, 9,
                   "Lecture : « Effet » = évolution moyenne des bénéficiaires entre M et "
                   "l'horizon, moins celle de leurs témoins appariés (même mois de RDV, profil "
                   "comparable). Un IC à 95 % qui ne contient pas 0 indique un effet significatif. "
                   "Qualité de l'appariement : AUC du score de propension = "
                   f"{infos['auc']:.2f} ; toutes les SMD après appariement < 0,10 = équilibre "
                   "satisfaisant (voir feuille Matching).", cl.f_sous)
    ws.set_row(fin + 2, 42)

    # --- Qualité des données ------------------------------------------------
    ws = cl.feuille("Qualité données", "Contrôles qualité",
                    "Contrôles exécutés au chargement des deux tables.")
    q = pd.DataFrame(qualite, columns=["Contrôle", "Valeur", "Commentaire / action"])
    cl.table(ws, q, 4, 1, formats={"Commentaire / action": cl.f_txt},
             largeurs={"Contrôle": 42, "Commentaire / action": 55})

    # --- Descriptif ---------------------------------------------------------
    ws = cl.feuille("Descriptif", "Profil des deux populations",
                    "Avant appariement — l'écart entre colonnes illustre le biais de "
                    "sélection que le matching corrige.")
    fin = cl.table(ws, profil, 4, 1,
                   formats={"Bénéficiaires": cl.f_dec, "Témoins": cl.f_dec,
                            "Écart": cl.f_dec},
                   largeurs={"Indicateur": 34})
    eff = effectifs.reset_index().rename(columns={COL_MOIS: "Mois de RDV"})
    cl.table(ws, eff, 4, 7)
    cl.image(ws, fin + 2, 1, DOSSIER_FIG / "fig_effectifs.png")

    # --- Trajectoires -------------------------------------------------------
    ws = cl.feuille("Trajectoires", "Trajectoires moyennes M-6 → M+6",
                    "Échantillon apparié : les courbes pré-RDV quasi parallèles valident la "
                    "comparaison ; l'écart qui s'ouvre après M mesure visuellement l'effet.")
    cl.image(ws, 4, 1, DOSSIER_FIG / "fig_trajectoires.png")

    # --- Cohortes -----------------------------------------------------------
    ws = cl.feuille("Cohortes", "Analyse de cohortes par mois de RDV",
                    "Indice base 100 au mois du RDV. « n.d. » : horizon non encore observable "
                    "(données arrêtées à décembre 2025).")
    cl.image(ws, 4, 1, DOSSIER_FIG / "fig_cohortes_heatmap.png")
    lig = 32
    for stem, tab in indices.items():
        ws.write(lig, 1, f"{stem} — indice base 100 à M", cl.f_h2)
        t = tab.round(1).reset_index().rename(columns={COL_MOIS: "Cohorte"})
        lig = cl.table(ws, t, lig + 1, 1,
                       formats={c: cl.f_dec for c in t.columns if c != "Cohorte"}) + 2

    # --- Matching -----------------------------------------------------------
    ws = cl.feuille("Matching", "Appariement par score de propension",
                    f"Modèle logistique (AUC = {infos['auc']:.2f}), appariement 1:1 sans "
                    f"remise, exact sur le mois de RDV, caliper = {infos['caliper']:.2f} sur le "
                    f"logit. {infos['n_apparies']} paires formées "
                    f"({infos['taux']:.0f} % des bénéficiaires).")
    cl.image(ws, 4, 1, DOSSIER_FIG / "fig_balance.png")
    s = smd.round(3)
    ws.write(4, 11, "Différences moyennes standardisées", cl.f_h2)
    cl.table(ws, s, 5, 11, formats={"SMD avant": cl.f_dec2, "SMD après": cl.f_dec2},
             largeurs={"Variable": 38})

    # --- Impact DiD ---------------------------------------------------------
    ws = cl.feuille("Impact (DiD)", "Effet net par variable et horizon",
                    "Différence de différences sur paires appariées ; IC à 95 % ; "
                    "* = significatif à 5 %.")
    cl.image(ws, 4, 1, DOSSIER_FIG / "fig_did.png")
    lig = 4 + 34
    d = did.copy()
    fin = cl.table(ws, d, lig, 1,
                   formats={"Effet": cl.f_dec, "IC95 bas": cl.f_dec, "IC95 haut": cl.f_dec,
                            "Effet relatif (%)": cl.f_dec2, "p-value": cl.f_pval,
                            "N paires": cl.f_int})
    for i, (_, r) in enumerate(d.iterrows()):
        fmt = cl.f_sig if r["Significatif (5 %)"] == "Oui" else cl.f_nsig
        ws.write(lig + 1 + i, 1 + len(d.columns) - 1, r["Significatif (5 %)"], fmt)

    cl.wb.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    SORTIE.mkdir(exist_ok=True)
    DOSSIER_FIG.mkdir(exist_ok=True)

    print("1/6 Chargement et contrôles qualité…")
    traites, temoins, stems, qualite = charger()
    print(f"    {len(traites)} bénéficiaires, {len(temoins)} témoins, "
          f"{len(stems)} variables d'évolution : {', '.join(stems)}")

    print("2/6 Descriptif…")
    profil, effectifs = descriptif(traites, temoins, stems)

    print("3/6 Cohortes…")
    moyennes, indices = analyse_cohortes(traites, stems)

    print("4/6 Appariement par score de propension…")
    apparie_t, apparie_c, smd, infos = apparier(traites, temoins, stems)
    print(f"    AUC = {infos['auc']:.3f} ; {infos['n_apparies']} paires "
          f"({infos['taux']:.1f} % des bénéficiaires) ; "
          f"SMD max après appariement = {smd['SMD après'].max():.3f}")

    print("5/6 Impact (différence de différences)…")
    did = impact_did(apparie_t, apparie_c, stems)
    n_sig = int((did["p-value"] < 0.05).sum())
    print(f"    {len(did)} effets estimés, {n_sig} significatifs à 5 %")

    print("6/6 Figures et export Excel…")
    stem_heatmap = "ENCOURS_EPARGNE" if "ENCOURS_EPARGNE" in stems else stems[0]
    fig_effectifs(effectifs)
    fig_trajectoires(apparie_t, apparie_c, stems)
    fig_heatmap_cohortes(indices, stem_heatmap)
    fig_did(did)
    fig_balance(smd)
    exporter_excel(qualite, profil, effectifs, moyennes, indices, stem_heatmap,
                   smd, infos, did)
    print(f"Terminé : {FICHIER_EXCEL}")


if __name__ == "__main__":
    main()
