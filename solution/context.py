"""Evidencia de equipo SIN balón, a partir de la forma y el movimiento de los dos equipos.

El balón falta en ~41% de los frames puntuables, en huecos de ~10 s de mediana. Ahí la única pista
es cómo están colocados y cómo se mueven los 22: el equipo con el balón se estira y avanza, el otro
se junta y recula.

Cómo se aprende sin etiquetas humanas (permitido por RULES.md: pseudo-etiquetas algorítmicas):
  1. Frames "seguros": el balón se ve, un jugador está a <1.5 m y la decodificación previa (v2) da
     posesión a ese mismo equipo.
  2. Se entrena una regresión logística SOLO con rasgos de jugadores (sin balón) para predecir ese
     equipo.
  3. Se aplica en los frames donde no hay balón.
"""
import warnings

import numpy as np

FEATURES = ["dcent", "dfront", "dback", "dspread_x", "dspread_y", "dvel", "hvel", "avel"]


def attack_sign(players, team, halves):
    """+1 si el local ataca hacia x creciente en esa parte, -1 si no. Por mayoría de posiciones."""
    sign = np.zeros(len(players))
    for _, s, e in halves:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            hx = np.nanmean(players[s:e + 1][:, team == 0, 0])
            ax = np.nanmean(players[s:e + 1][:, team == 1, 0])
        sign[s:e + 1] = 1.0 if hx < ax else -1.0   # el local defiende su lado: ataca hacia el otro
    return sign


def _smooth(x, w):
    ok = ~np.isnan(x)
    k = np.ones(2 * w + 1)
    num = np.convolve(np.where(ok, x, 0.0), k, "same")
    den = np.convolve(ok.astype(float), k, "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def player_features(atd, halves):
    """Matriz (N, len(FEATURES)) en perspectiva del LOCAL atacando hacia +x."""
    players, team = atd["players"], atd["team"]
    L = atd["pitch"][0]
    s = attack_sign(players, team, halves)
    x = (players[:, :, 0] - L / 2) * s[:, None]   # centrado y orientado
    y = players[:, :, 1]
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stats = {}
        for t, name in ((0, "h"), (1, "a")):
            xt, yt = x[:, team == t], y[:, team == t]
            stats[name] = dict(c=np.nanmean(xt, 1), front=np.nanmax(xt, 1),
                               # el más retrasado suele ser el portero: usamos el 2º más retrasado
                               back=np.sort(np.where(np.isnan(xt), np.inf, xt), 1)[:, 1],
                               sx=np.nanstd(xt, 1), sy=np.nanstd(yt, 1))
            stats[name]["back"][np.isinf(stats[name]["back"])] = np.nan
            stats[name]["v"] = _smooth(np.gradient(_smooth(stats[name]["c"], 12)) * 25, 12)
    h, a = stats["h"], stats["a"]
    out["dcent"] = h["c"] + a["c"]        # a ve el campo invertido: suma = cuánto empuja el local
    out["dfront"] = h["front"] + a["back"]
    out["dback"] = h["back"] + a["front"]
    out["dspread_x"] = h["sx"] - a["sx"]
    out["dspread_y"] = h["sy"] - a["sy"]
    out["dvel"] = h["v"] - a["v"]
    out["hvel"] = h["v"]
    out["avel"] = a["v"]
    return np.column_stack([out[k] for k in FEATURES])


def fit_logistic(X, y, l2=1e-2, iters=300, lr=0.5):
    """Regresión logística con descenso de gradiente (numpy puro, determinista)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]]
        w -= lr * g
    return dict(w=w, mu=mu, sd=sd)


def logit(model, X):
    Z = np.column_stack([np.ones(len(X)), (X - model["mu"]) / model["sd"]])
    return Z @ model["w"]


def confident_frames(f, labels, max_dist=1.5):
    """Pseudo-etiquetas: 1 = local, 0 = visitante, -1 = no usar."""
    dh = np.nan_to_num(f["d_home"], nan=99)
    da = np.nan_to_num(f["d_away"], nan=99)
    y = np.full(len(labels), -1)
    near_h = (dh < max_dist) & (dh < da) & (labels == "H")
    near_a = (da < max_dist) & (da < dh) & (labels == "A")
    y[near_h], y[near_a] = 1, 0
    return y
