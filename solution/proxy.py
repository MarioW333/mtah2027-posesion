"""spellF1 SIN ground truth: pseudo-verdad + balón borrado + el scorer oficial.

Pseudo-verdad (solo donde se ve el balón):
  - equipo = el del jugador a < OWN_DIST del balón; si nadie, se mantiene el último (regla del
    reto: "en fases disputadas la posesión sigue con el último que la tuvo")
  - D = lo que decide el modelo de referencia (v7)
  - X (no puntuado) dentro de cualquier hueco real del balón de más de 1 s: ahí no sabemos nada
Después se BORRA el balón en tramos adicionales (con longitudes sacadas de los huecos reales) y se
mide cuánto se aleja cada configuración de la pseudo-verdad, con possession_metrics.spell_scores.

Límite conocido: la pseudo-verdad sale del mismo tracking ruidoso, así que no simula los errores
del tracker -- solo la falta de balón. Mide robustez, no exactitud absoluta.

    python solution/proxy.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from games import half_video_blocks  # noqa: E402
from solution.load import load_atd, load_set_pieces  # noqa: E402
from solution.model import P, features, predict  # noqa: E402

OWN_DIST = 1.0
MASK_FRACTION = 0.3
SEED = 1


def pseudo_truth(atd, halves, ref_labels):
    f = features(atd)
    raw_ok = ~np.isnan(atd["ball"][:, 0])
    dh = np.nan_to_num(f["d_home"], nan=99)
    da = np.nan_to_num(f["d_away"], nan=99)
    owner = np.where(raw_ok & (np.minimum(dh, da) < OWN_DIST), np.where(dh < da, 0, 1), -1)
    lab = np.full(len(owner), "X", dtype="<U1")
    for _, s, e in halves:
        last = -1
        for t in range(s, e + 1):
            if owner[t] >= 0:
                last = owner[t]
            lab[t] = "X" if last < 0 else "HA"[last]
    lab[ref_labels == "D"] = "D"
    # huecos reales > 1 s: verdad desconocida
    gap = ~raw_ok
    run = np.zeros(len(gap), int)
    for t in range(1, len(gap)):
        run[t] = run[t - 1] + 1 if gap[t] else 0
    long_gap = np.zeros(len(gap), bool)
    for t in np.flatnonzero((run > 25) & ~np.r_[gap[1:], False]):
        long_gap[t - run[t] + 1:t + 1] = True
    lab[long_gap & (lab != "D")] = "X"
    return lab


def mask_ball(atd, halves, seed=SEED, fraction=MASK_FRACTION):
    """Borra el balón en tramos aleatorios con la distribución de longitudes de los huecos reales."""
    rng = np.random.default_rng(seed)
    ok = ~np.isnan(atd["ball"][:, 0])
    edges = np.diff(np.r_[0, (~ok).astype(np.int8), 0])
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    lengths = lengths[lengths > 25]
    override = np.ones(len(ok), bool)
    for _, s, e in halves:
        target = fraction * ok[s:e + 1].sum()
        hidden = 0
        while hidden < target:
            L = int(rng.choice(lengths))
            a = int(rng.integers(s, e + 1 - L))
            hidden += np.sum(override[a:a + L] & ok[a:a + L])
            override[a:a + L] = False
    return dict(atd, ball_override=override)


def score(truth, pred):
    gt = pm.label_codes(truth)
    pr = pm.label_codes(pred)
    runs = pm.valid_runs(gt)
    sp = pm.spell_scores(gt, pm.fill_abstentions(pr, runs), runs)
    cm = pm.confusion(gt, pr)
    return sp["mean_f1"], pm.macro_f1(pm.scored_confusion(cm)), sp["n_gt"], sp["n_pred"]


def run(configs):
    atd = load_atd()
    halves = half_video_blocks("BAR-ATH")
    sp = load_set_pieces()
    ref, _ = predict(atd, halves, sp, P)
    truth = pseudo_truth(atd, halves, ref)
    masked = mask_ball(atd, halves)
    print("pseudo-verdad: %d frames puntuados" % np.sum(truth != "X"))
    for name, over in configs:
        lab, _ = predict(masked, halves, sp, dict(P, **over))
        s, m, ng, npred = score(truth, lab)
        print("  %-28s spellF1 %.4f | macroF1 %.4f | posesiones verdad %d, pred %d"
              % (name, s, m, ng, npred), flush=True)


if __name__ == "__main__":
    run([("v7 actual", {})])
