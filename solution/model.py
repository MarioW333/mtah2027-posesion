"""Posesión por frame con un modelo de 3 estados (H, A, D) decodificado con Viterbi.

Idea: cada frame aporta evidencia débil y ruidosa (quién está más cerca del balón, si el juego
parece parado). Viterbi elige la secuencia de estados que mejor explica esa evidencia pagando una
penalización fija por cada cambio de estado. Esa penalización es lo que impide el parpadeo y
controla cuántas posesiones salen -- exactamente lo que mide spellF1.

Sin etiquetas: todos los parámetros son constantes físicas/futbolísticas razonadas, no ajustadas
contra el ground truth.
"""
import numpy as np

H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])

# ---------------------------------------------------------------- parámetros (razonados, no ajustados)
P = dict(
    interp_gap=12,        # frames: huecos del balón <= 0.5 s se interpolan linealmente
    control_dist=2.0,     # m: a esta distancia un jugador "tiene" el balón
    team_gain=1.0,        # evidencia máxima H-vs-A por frame (log-odds)
    smooth_speed=25,      # frames: ventana para velocidades (1 s)
    walk_speed=1.2,       # m/s: velocidad mediana de jugadores típica de una parada
    dead_gain=1.0,        # log-odds D por cada m/s por debajo de walk_speed (recortado a +-1)
    still_ball=0.5,       # m/s: balón prácticamente quieto
    still_bonus=1.0,      # log-odds D extra si el balón está quieto
    outside_bonus=3.0,    # log-odds D si el balón está fuera del campo
    missing_after=25,     # frames: un balón perdido más de 1 s sugiere parada (en manos, fuera)
    missing_bonus=0.8,    # log-odds D en esos frames
    tag_before=100,       # frames: antes del inicio de un SET PIECE el balón está parado (4 s)
    tag_bonus=1.5,        # log-odds D en esa ventana
    target_dead=0.38,     # prior futbolístico: ~38% del tiempo el balón no está en juego
    switch_team=10.0,     # coste de un cambio H<->A (~0.5 s de control claro)
    switch_dead=15.0,     # coste de entrar/salir de D
    use_context=True,     # evidencia de equipo sin balón (context.py)
    ctx_frames=10,        # peso de la evidencia de contexto (elegido con verify.py, sin GT)
    use_dead_clf=True,    # detector de paradas aprendido (dead.py)
    dead_clf_weight=0.5,  # peso de su logit por frame
    dead_ball_view=True,  # segunda vista (rasgos del balón) en el detector de paradas
    use_tags=True,        # equipo según las fases del Smart Tagging (tags.py)
    tag_frames=25,        # peso: una fase cuenta como 1 observación por segundo
    use_control=True,     # episodios de control: los toques breves no cuentan
    touch_min=2,          # frames (0.08 s): por debajo, toque sin control -> peso 0
    control_full=12,      # frames (0.5 s): a partir de aquí, control claro -> peso 1
                          # (laxo a propósito: el reto cuenta posesiones de <1 s; elegido con proxy.py)
    alive_in_dead_max=0,  # frames: posesiones más cortas entre dos D -> D (0 = desactivado)
)


def _interp_short_gaps(x, max_gap):
    """Interpola linealmente huecos NaN de longitud <= max_gap en un array 1D."""
    x = x.copy()
    nan = np.isnan(x)
    if nan.all():
        return x
    idx = np.arange(len(x))
    filled = np.interp(idx, idx[~nan], x[~nan])
    # longitud de cada hueco
    edges = np.diff(np.concatenate(([0], nan.view(np.int8), [0])))
    for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        if e - s <= max_gap and s > 0 and e < len(x):
            x[s:e] = filled[s:e]
    return x


def _moving_mean(x, w):
    """Media móvil centrada ignorando NaN."""
    ok = ~np.isnan(x)
    k = np.ones(2 * w + 1)
    num = np.convolve(np.where(ok, x, 0.0), k, "same")
    den = np.convolve(ok.astype(float), k, "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def features(atd, p=P):
    players, ball, team = atd["players"], atd["ball"], atd["team"]
    L, W = atd["pitch"]

    bx = _interp_short_gaps(ball[:, 0].astype(float), p["interp_gap"])
    by = _interp_short_gaps(ball[:, 1].astype(float), p["interp_gap"])
    ball_ok = ~np.isnan(bx)

    # distancia del balón al jugador más cercano de cada equipo
    dist = np.hypot(players[:, :, 0] - bx[:, None], players[:, :, 1] - by[:, None])
    with np.errstate(invalid="ignore"):
        d_home = np.nanmin(np.where(team[None] == 0, dist, np.nan), axis=1)
        d_away = np.nanmin(np.where(team[None] == 1, dist, np.nan), axis=1)

    # velocidad del balón
    bs = np.hypot(np.diff(bx, prepend=np.nan), np.diff(by, prepend=np.nan)) * 25
    ball_speed = _moving_mean(bs, p["smooth_speed"] // 5)

    # velocidad mediana de los jugadores (lo "activo" que está el partido)
    v = np.linalg.norm(np.diff(players, axis=0, prepend=np.nan), axis=2) * 25
    with np.errstate(invalid="ignore"):
        v[v > 12] = np.nan                       # saltos de identidad del tracker, no carreras
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            play_speed = np.nanmedian(v, axis=1)
    play_speed = _moving_mean(play_speed, p["smooth_speed"])

    m = 0.5
    outside = ball_ok & ((bx < -m) | (bx > L + m) | (by < -m) | (by > W + m))
    return dict(ball_ok=ball_ok, d_home=d_home, d_away=d_away, ball_speed=ball_speed,
                play_speed=play_speed, outside=outside,
                control_w=control_weight(d_home, d_away, p))


def control_weight(d_home, d_away, p=P):
    """Peso de cada frame según la duración del EPISODIO de control al que pertenece.

    Episodio = racha de frames seguidos en que el mismo equipo tiene un jugador a < control_dist.
    "Un simple toque sin control no cuenta" (CHALLENGE.md): un episodio de menos de touch_min
    frames pesa 0 (despeje, rebote, desvío); a partir de control_full frames pesa 1; lineal entre
    medias. No es causal -- Viterbi decide mirando toda la secuencia, así que puede usar la
    duración completa del episodio.
    """
    dh = np.nan_to_num(d_home, nan=99.0)
    da = np.nan_to_num(d_away, nan=99.0)
    who = np.where(np.minimum(dh, da) < p["control_dist"], np.where(dh < da, 0, 1), -1)
    w = np.ones(len(who))
    change = np.flatnonzero(np.diff(who)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(who)]))
    for s, e in zip(starts, ends):
        if who[s] >= 0:
            w[s:e] = np.clip((e - s - p["touch_min"]) / (p["control_full"] - p["touch_min"]), 0, 1)
    return w


def _run_length(mask):
    """Para cada frame True, cuántos frames seguidos lleva True (incluido él)."""
    out = np.zeros(len(mask), np.int64)
    c = 0
    for i, m in enumerate(mask):
        c = c + 1 if m else 0
        out[i] = c
    return out


def dead_evidence(f, set_pieces, p=P):
    """Log-odds de balón muerto por frame, SIN el sesgo global (lo calibra predict)."""
    n = len(f["ball_ok"])
    if "dead_clf" in f:       # detector aprendido (dead.py) en lugar de la regla de velocidad
        dead = np.clip(f["dead_clf"], -4, 4) * p["dead_clf_weight"]
    else:
        ps = np.nan_to_num(f["play_speed"], nan=p["walk_speed"])
        dead = np.clip(p["dead_gain"] * (p["walk_speed"] - ps), -1, 1)
    still = f["ball_ok"] & (np.nan_to_num(f["ball_speed"], nan=9.0) < p["still_ball"])
    dead += p["still_bonus"] * still
    dead += p["outside_bonus"] * f["outside"]
    dead += p["missing_bonus"] * (_run_length(~f["ball_ok"]) > p["missing_after"])
    for start, _, _ in set_pieces:
        dead[max(0, start - p["tag_before"]):min(n, start)] += p["tag_bonus"]
    return dead


def emissions(f, dead, p=P):
    """Log-puntuaciones (N,3) para H, A, D."""
    n = len(f["ball_ok"])

    # --- evidencia de equipo: quién está más cerca, pesada por lo cerca que está
    dh = np.nan_to_num(f["d_home"], nan=99.0)
    da = np.nan_to_num(f["d_away"], nan=99.0)
    closest = np.minimum(dh, da)
    weight = np.exp(-np.maximum(closest - p["control_dist"], 0) / p["control_dist"])
    if p["use_control"]:
        weight = weight * f["control_w"]
    team = p["team_gain"] * weight * np.tanh((da - dh) / 2.0)
    team[~f["ball_ok"]] = 0.0
    if "ctx" in f:
        # sin balón: forma de los equipos (context.py). Log-odds de verosimilitud, contadas como
        # ~1 observación independiente por segundo para no sobrecontar frames correlacionados.
        gap = ~f["ball_ok"]
        team[gap] = np.clip(np.nan_to_num(f["ctx"][gap]), -3, 3) / p["ctx_frames"]
    if "tag_team" in f:
        # Smart Tagging: log-odds por fase, repartidas por frame (1 observación cada tag_frames)
        team = team + f["tag_team"] / p["tag_frames"]

    e = np.zeros((n, 3))
    e[:, H] = team / 2 - dead / 2
    e[:, A] = -team / 2 - dead / 2
    # Con el balón muerto, quién está cerca no dice nada: la pista de equipo debe puntuar en D
    # igual que la mezcla 50/50 de H y A. Sin esto, "un jugador junto al balón" cuenta como
    # prueba de balón vivo y tapa las paradas (p. ej. un saque de banda con el balón en la mano).
    e[:, D] = dead / 2 + np.logaddexp(team / 2, -team / 2) - np.log(2)
    return e


def viterbi(e, p=P):
    """Secuencia de estados de máxima puntuación con costes de cambio fijos."""
    n = len(e)
    cost = np.array([[0, p["switch_team"], p["switch_dead"]],
                     [p["switch_team"], 0, p["switch_dead"]],
                     [p["switch_dead"], p["switch_dead"], 0]])
    score = e[0].copy()
    back = np.zeros((n, 3), np.int8)
    for t in range(1, n):
        cand = score[:, None] - cost          # cand[i, j]: venir de i y pasar a j
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + e[t]
    path = np.empty(n, np.int8)
    path[-1] = np.argmax(score)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _decode(e, halves, p):
    labels = np.full(len(e), "X", dtype="<U1")
    for _, start, end in halves:
        labels[start:end + 1] = LABELS[viterbi(e[start:end + 1], p)]
    return postprocess(labels, p)


def postprocess(labels, p=P):
    """Una posesión de menos de p["alive_in_dead_max"] frames entre dos paradas pasa a D: en
    medio de una parada no hay tiempo para una posesión real (suele ser el balón devuelto)."""
    k = p["alive_in_dead_max"]
    if k <= 0:
        return labels
    out = labels.copy()
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(labels)]
    for i in range(1, len(starts) - 1):
        if (labels[starts[i]] in ("H", "A") and ends[i] - starts[i] < k
                and labels[starts[i - 1]] == "D" and labels[starts[i + 1]] == "D"):
            out[starts[i]:ends[i]] = "D"
    return out


def predict(atd, halves, set_pieces=(), p=P):
    """Etiquetas para todo el vídeo. Fuera de las partes: X.

    El sesgo global vivo/muerto se calibra sin etiquetas: búsqueda binaria hasta que la fracción
    de D decodificada en todo el partido iguala p["target_dead"] (un prior futbolístico general,
    no un dato de este partido).
    """
    f = features(atd, p)
    if "ball_override" in atd:            # para verify.py: huecos de balón simulados
        f["ball_ok"] &= atd["ball_override"]
    graded = np.zeros(len(f["ball_ok"]), bool)
    for _, s, e_ in halves:
        graded[s:e_ + 1] = True
    if p["use_dead_clf"]:
        from solution.dead import dead_classifier
        f["dead_clf"], _ = dead_classifier(atd, f, set_pieces, graded,
                                           ball_view=p["dead_ball_view"])
    raw = dead_evidence(f, set_pieces, p)

    labels, f["dead_bias"] = _calibrated_decode(f, raw, halves, graded, p)
    if p["use_tags"]:
        from solution.context import confident_frames
        from solution.tags import load_phases, tag_evidence
        f["tag_team"], f["tag_reliab"] = tag_evidence(load_phases(), confident_frames(f, labels),
                                                      len(labels))
    if p["use_context"]:
        from solution.context import (confident_frames, fit_logistic, logit,
                                      player_features)
        X = player_features(atd, halves)
        y = confident_frames(f, labels)
        ok = (y >= 0) & graded & ~np.isnan(X).any(1)
        model = fit_logistic(X[ok], y[ok])
        prior = np.log(y[ok].mean() / (1 - y[ok].mean()))
        f["ctx"] = logit(model, np.nan_to_num(X)) - prior    # verosimilitud, sin el prior de clase
        f["ctx_model"] = model
    if p["use_context"] or p["use_tags"]:
        labels, f["dead_bias"] = _calibrated_decode(f, raw, halves, graded, p)
    return labels, f


def _calibrated_decode(f, raw, halves, graded, p):
    """Búsqueda binaria del sesgo vivo/muerto hasta que la fracción de D = target_dead."""
    lo, hi = -3.0, 3.0
    for _ in range(12):
        bias = (lo + hi) / 2
        labels = _decode(emissions(f, raw + bias, p), halves, p)
        if np.mean(labels[graded] == "D") < p["target_dead"]:
            lo = bias
        else:
            hi = bias
    return labels, bias
