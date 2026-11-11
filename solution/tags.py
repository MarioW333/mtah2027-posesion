"""Smart Tagging como evidencia débil de QUÉ EQUIPO tiene el balón.

Cada fase de juego lleva un `Team`. En las fases ofensivas (BUILD UP, PROGRESSION...) es el equipo
con el balón; en las defensivas (DEFENDING IN ..., DEFENSIVE TRANSITION) es el que defiende, así que
se invierte. La fiabilidad de cada código NO se supone: se mide contra pseudo-etiquetas (frames con
el balón pegado a un jugador) y se convierte en log-odds. Códigos cerca del 50% se ignoran.
"""
import xml.etree.ElementTree as ET

import numpy as np

from games import GAME, require

SKIP = {"SET PIECES", "GOALS"}       # el equipo de un saque/gol no es el de la posesión en curso


def load_phases(game=GAME):
    """Lista de (frame_ini, frame_fin, code, signo) con signo +1 Barcelona(local), -1 visitante."""
    path = require(["smart_tagging"], game)[0]
    out = []
    for inst in ET.parse(path).getroot().iter("instance"):
        code = (inst.findtext("code") or "").strip().upper()
        labels = {l.findtext("group"): (l.findtext("text") or "").strip().lower()
                  for l in inst.findall("label")}
        team = labels.get("Team", "")
        if code in SKIP or team not in ("barcelona", "atletic club"):
            continue
        out.append((int(round(float(inst.findtext("start")) * 25)),
                    int(round(float(inst.findtext("end")) * 25)), code,
                    1 if team == "barcelona" else -1))
    return out


def tag_evidence(phases, pseudo, n, min_margin=0.15):
    """Log-odds H-vs-A por frame (suma de las fases que lo cubren) y la fiabilidad por código.

    pseudo: 1 local, 0 visitante, -1 sin pseudo-etiqueta.
    """
    reliab = {}
    for code in sorted({p[2] for p in phases}):
        hit = tot = 0
        for s, e, c, sign in phases:
            if c != code:
                continue
            y = pseudo[s:e + 1]
            y = y[y >= 0]
            hit += np.sum(y == (1 if sign > 0 else 0))
            tot += len(y)
        if tot >= 500:
            reliab[code] = (hit + 1) / (tot + 2)          # suavizado de Laplace
    ev = np.zeros(n)
    for s, e, code, sign in phases:
        r = reliab.get(code)
        if r is None or abs(r - 0.5) < min_margin:
            continue
        ev[s:e + 1] += sign * np.log(r / (1 - r))       # r<0.5 invierte el signo solo
    return ev, reliab
