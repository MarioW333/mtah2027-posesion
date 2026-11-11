"""Detector de balón muerto aprendido sin etiquetas humanas (co-entrenamiento).

Las pseudo-etiquetas salen de reglas sobre el BALÓN y el Smart Tagging:
  muerto: balón fuera del campo, o los 4 s antes del inicio de un SET PIECES
  vivo:   balón visible dentro del campo, rápido (>6 m/s) y lejos de cualquier pseudo-muerto
El clasificador, en cambio, solo ve rasgos de los JUGADORES (velocidades, frenazos, colocación).
Como no ve las reglas que crearon las etiquetas, tiene que aprender la firma colectiva de una
parada -- y así la detecta también donde las reglas no llegan (balón llevado en la mano a la banda,
falta en mitad del campo...).
"""
import warnings

import numpy as np

from solution.context import fit_logistic, logit

FEATURES = ["v1", "v_past3", "v_next3", "v_drop", "v_rise", "slow_frac", "spread", "n_vis"]


def _mean(x, a, b):
    """Media de x en la ventana [t+a, t+b] (frames), ignorando NaN."""
    ok = ~np.isnan(x)
    cs = np.concatenate(([0], np.cumsum(np.where(ok, x, 0.0))))
    cn = np.concatenate(([0], np.cumsum(ok)))
    t = np.arange(len(x))
    lo = np.clip(t + a, 0, len(x))
    hi = np.clip(t + b + 1, 0, len(x))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (cs[hi] - cs[lo]) / (cn[hi] - cn[lo])


def player_activity(atd):
    players = atd["players"]
    v = np.linalg.norm(np.diff(players, axis=0, prepend=np.nan), axis=2) * 25
    v[v > 12] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(v, 1)
        slow = np.nanmean(np.where(np.isnan(v), np.nan, v < 1.5), 1)
        spread = np.nanstd(players[:, :, 0], 1) + np.nanstd(players[:, :, 1], 1)
    n_vis = np.sum(~np.isnan(players[:, :, 0]), 1).astype(float)
    v1 = _mean(med, -12, 12)
    past, nxt = _mean(med, -75, -1), _mean(med, 1, 75)
    X = np.column_stack([v1, past, nxt, past - v1, nxt - v1, _mean(slow, -25, 25),
                         _mean(spread, -25, 25), n_vis])
    return X


def pseudo_dead(f, set_pieces, n, before=100, margin=3.0):
    """1 muerto, 0 vivo, -1 sin etiqueta."""
    y = np.full(n, -1)
    dead = f["outside"].copy()
    for s, _, _ in set_pieces:
        dead[max(0, s - before):min(n, s)] = True
    near_dead = _mean(dead.astype(float), -250, 250) > 0       # +-10 s de cualquier muerto
    fast = f["ball_ok"] & (np.nan_to_num(f["ball_speed"]) > 6) & ~f["outside"]
    y[fast & ~near_dead] = 0
    y[dead] = 1
    return y


BALL_FEATURES = ["line_dist", "spd1", "spd_past3", "spd_next3", "disp3", "near1"]


def ball_activity(atd, f):
    """Rasgos del BALÓN (segunda vista): distancia a la línea, velocidades en ventanas,
    desplazamiento en 3 s (llevado despacio = poco) y cercanía de un jugador.
    Sin visibilidad a propósito: que no se vea el balón ya lo cuenta model.dead_evidence, y aquí
    convertía cada hueco de juego vivo en una parada."""
    L, W = atd["pitch"]
    b = atd["ball"].astype(float)
    line = np.minimum.reduce([b[:, 0], L - b[:, 0], b[:, 1], W - b[:, 1]])
    spd = np.where(f["ball_ok"], f["ball_speed"], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sp_past = np.nanmax(np.lib.stride_tricks.sliding_window_view(
            np.concatenate((np.full(75, np.nan), spd)), 75)[:-1], 1)
        sp_next = np.nanmax(np.lib.stride_tricks.sliding_window_view(
            np.concatenate((spd, np.full(75, np.nan))), 75)[1:], 1)
    bx, by = b[:, 0], b[:, 1]
    disp = np.hypot(np.roll(bx, -37) - np.roll(bx, 37), np.roll(by, -37) - np.roll(by, 37))
    near = np.minimum(np.nan_to_num(f["d_home"], nan=99), np.nan_to_num(f["d_away"], nan=99))
    near = np.where(f["ball_ok"], np.minimum(near, 20), np.nan)
    return np.column_stack([np.clip(line, -5, 20), _mean(spd, -25, 25), sp_past, sp_next, disp,
                            _mean(near, -25, 25)])


def _fit(X, y, mask):
    ok = mask & (y >= 0)
    fill = np.nanmean(X[ok], 0)
    Xf = np.where(np.isnan(X), fill, X)
    model = fit_logistic(Xf[ok], y[ok])
    prior = np.log(y[ok].mean() / (1 - y[ok].mean()))
    return logit(model, Xf) - prior, model


def dead_classifier(atd, f, set_pieces, graded, confident=2.0, ball_view=True):
    """Co-entrenamiento en dos vistas.

    Vista 1 (jugadores): aprende de reglas sobre balón + Smart Tagging.
    Vista 2 (balón): aprende de las reglas + los frames donde la vista 1 está muy segura.
    Las dos vistas ven cosas distintas, así que sus log-odds se suman (Bayes ingenuo).
    """
    X1 = player_activity(atd)
    y = pseudo_dead(f, set_pieces, len(X1))
    z1, m1 = _fit(X1, y, graded)
    if not ball_view:
        return z1, dict(z_players=z1, model_players=m1, y=y)

    X2 = ball_activity(atd, f)
    y2 = y.copy()
    y2[(y2 < 0) & (z1 > confident)] = 1
    y2[(y2 < 0) & (z1 < -confident)] = 0
    z2, m2 = _fit(X2, y2, graded & f["ball_ok"])
    z2[~f["ball_ok"]] = 0.0          # sin balón, esta vista no sabe nada: se abstiene
    return z1 + z2, dict(z_players=z1, z_ball=z2, model_players=m1, model_ball=m2, y=y, y2=y2)
