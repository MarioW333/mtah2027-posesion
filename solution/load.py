"""Carga el ATD táctico en arrays numpy (con caché en data/cache, que está en .gitignore)."""
import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

N_SLOTS = 46


def team_of_slot(meta_path):
    """0 = home, 1 = away, por cada casilla de track (primer equipo del XML = local)."""
    root = ET.parse(meta_path).getroot()
    home_id = [t.get("id") for t in root.iter("Team")][0]
    return np.array([0 if p.get("teamId") == home_id else 1 for p in root.iter("Player")], np.int8)


def load_atd(game=GAME, use_cache=True):
    """Devuelve dict con:
    players (N,46,2) metros, NaN si no visible;  ball (N,2) metros;  team (46,) 0/1;  pitch (L,W)
    El índice es el frame de vídeo (0-based).
    """
    cache = os.path.join(ROOT, "data", "cache", "{}_atd.npz".format(game))
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    L, W = GAMES[game]["pitch"]
    if use_cache and os.path.exists(cache):
        d = np.load(cache)
        return dict(players=d["players"], ball=d["ball"], team=d["team"], pitch=(L, W))

    with open(raw_path) as f:
        text = f.read().replace(":", ",").replace(";", ",")
    arr = pd.read_csv(io.StringIO(text), header=None, dtype=np.float32, na_values=["NaN"]).values
    n = source_frames("tactical", game)
    frame = arr[:, 0].astype(int) - 1
    pl = arr[:, 1:1 + 3 * N_SLOTS].reshape(-1, N_SLOTS, 3)[:, :, :2]
    bl = arr[:, 1 + 3 * N_SLOTS:1 + 3 * N_SLOTS + 2]

    players = np.full((n, N_SLOTS, 2), np.nan, np.float32)
    ball = np.full((n, 2), np.nan, np.float32)
    players[frame] = pl * np.array([L, W], np.float32)
    ball[frame] = bl * np.array([L, W], np.float32)
    team = team_of_slot(meta_path)

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, players=players, ball=ball, team=team)
    return dict(players=players, ball=ball, team=team, pitch=(L, W))


def load_set_pieces(game=GAME):
    """Inicios de SET PIECES del Smart Tagging como frames de vídeo: lista de (frame_ini, frame_fin, tipo)."""
    path = require(["smart_tagging"], game)[0]
    out = []
    for inst in ET.parse(path).getroot().iter("instance"):
        if (inst.findtext("code") or "").strip().upper() != "SET PIECES":
            continue
        labels = {l.findtext("group"): l.findtext("text") for l in inst.findall("label")}
        out.append((int(round(float(inst.findtext("start")) * 25)),
                    int(round(float(inst.findtext("end")) * 25)), labels.get("Type")))
    return out
