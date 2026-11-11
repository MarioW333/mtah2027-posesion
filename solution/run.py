"""Genera la submission.   python solution/run.py [--name equipo]"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import half_video_blocks  # noqa: E402
from solution.load import load_atd, load_set_pieces  # noqa: E402
from solution.model import predict  # noqa: E402


def possessions(labels):
    """Número y duración media (s) de rachas H/A en los frames etiquetados."""
    lab = labels[labels != "X"]
    change = np.concatenate(([True], lab[1:] != lab[:-1]))
    starts = lab[change]
    runs = np.diff(np.concatenate((np.where(change)[0], [len(lab)])))
    alive = np.isin(starts, ["H", "A"])
    return int(alive.sum()), float(runs[alive].mean() / 25)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="hmm_v1")
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results"))
    a = ap.parse_args()

    t = time.time()
    atd = load_atd()
    labels, f = predict(atd, half_video_blocks("BAR-ATH"), load_set_pieces())
    print("sesgo vivo/muerto calibrado: %.2f" % f["dead_bias"])
    path = os.path.join(a.out, a.name + ".csv")
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))

    n, dur = possessions(labels)
    share = {c: round(float(np.mean(labels[labels != "X"] == c)), 3) for c in "HAD"}
    print("escrito", os.path.relpath(path, ROOT), "en %.0f s" % (time.time() - t))
    print("partido completo: %d posesiones, duración media %.1f s, reparto %s" % (n, dur, share))
