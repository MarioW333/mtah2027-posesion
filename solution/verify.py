"""Verificación SIN ground truth: ¿rellena bien los huecos del balón?

Se borra el balón a propósito en ventanas donde sí se ve (huecos simulados de 10 s, la mediana
real) y se compara lo que decide el modelo en esas ventanas con lo que decidía viendo el balón.
Así se mide la ayuda de la evidencia de contexto sin tocar las etiquetas públicas.

    python solution/verify.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import half_video_blocks  # noqa: E402
from solution.load import load_atd, load_set_pieces  # noqa: E402
from solution.model import P, predict  # noqa: E402

GAP, N_WIN, SEED = 250, 120, 0


BASE = dict(use_context=False, use_dead_clf=False, use_tags=False)   # v2: solo balón
VARIANTS = [
    ("v2 (solo balón)", BASE),
    ("v3 (+contexto)", dict(use_dead_clf=False, use_tags=False)),
    ("v4 (+paradas)", dict(use_tags=False)),
    ("v4 (+paradas+tags)", {}),
]


def main(variants=VARIANTS):
    atd = load_atd()
    halves = half_video_blocks("BAR-ATH")
    sp = load_set_pieces()
    # referencia: lo que dice SOLO el balón cuando se ve (sin contexto, paradas ni tags, para no
    # premiar a las variantes que comparten esas pistas con la referencia)
    ref, f = predict(atd, halves, sp, dict(P, **BASE))

    # ventanas candidatas: balón visible todo el rato y referencia viva (H/A) todo el rato
    good = f["ball_ok"] & np.isin(ref, ["H", "A"])
    rng = np.random.default_rng(SEED)
    starts, taken = [], np.zeros(len(ref), bool)
    for s in rng.permutation(np.where(good)[0]):
        if len(starts) == N_WIN:
            break
        w = slice(s, s + GAP)
        if s + GAP <= len(ref) and good[w].all() and not taken[max(0, s - GAP):s + 2 * GAP].any():
            starts.append(s)
            taken[w] = True
    print("ventanas simuladas: %d de %d frames (%.0f s)" % (len(starts), GAP, GAP / 25))

    masked = dict(atd, ball_override=~taken)
    for name, over in variants:
        lab, _ = predict(masked, halves, sp, dict(P, **over))
        agree = np.mean(lab[taken] == ref[taken])
        dead = np.mean(lab[taken] == "D")
        # ¿detecta los cambios de posesión que ocurren dentro del hueco?
        changes_ref = sum(np.sum(ref[s + 1:s + GAP] != ref[s:s + GAP - 1]) for s in starts)
        changes_hit = sum(np.sum((lab[s + 1:s + GAP] != lab[s:s + GAP - 1]) &
                                 np.isin(lab[s + 1:s + GAP], ["H", "A"])) for s in starts)
        print("  %-18s acuerdo con referencia %.3f | marca D %.3f | cambios dentro: ref %d, pred %d"
              % (name, agree, dead, changes_ref, changes_hit))


if __name__ == "__main__":
    main()
