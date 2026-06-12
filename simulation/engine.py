"""Vectorised Monte Carlo simulation of the 2026 FIFA World Cup.

Approach
--------
* Every possible fixture is predicted ONCE (batched through the model
  ensemble); simulation then only samples from cached distributions, which
  makes 100k tournament runs take seconds, not hours.
* Group matches are sampled as exact scorelines from the Dixon-Coles grid so
  group tables get realistic goal difference / goals-for tiebreakers.
* Knockout draws are resolved with an advancement probability
  P(adv) = p_win + p_draw * p_win / (p_win + p_loss)  (extra time / penalties
  resolve roughly in proportion to underlying strength).
* The FIFA third-place allocation (which best-third goes to which R32 match)
  is solved by backtracking per unique qualifying-set and memoised — there
  are at most C(12,8)=495 distinct cases.

Group tiebreakers are points → goal difference → goals scored → random,
a standard simulation approximation of the full FIFA criteria.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from itertools import combinations

import numpy as np

import config
from models.scoreline import outcome_probs_from_grid, score_grid
from simulation.worldcup2026 import GROUPS, HOSTS, R16, R32

logger = logging.getLogger(__name__)

GROUP_LETTERS = list(GROUPS)
N_GOALS = config.MAX_GOALS_GRID + 1


def _venue(home: str, away: str) -> str:
    if home in HOSTS:
        return HOSTS[home]
    if away in HOSTS:
        return HOSTS[away]
    return "United States"


class WorldCupSimulator:
    def __init__(self, predictor, seed: int = config.RANDOM_SEED):
        self.predictor = predictor
        self.rng = np.random.default_rng(seed)
        self.teams: list[str] = [t for g in GROUPS.values() for t in g]
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.kickoff = datetime(2026, 6, 11)
        self._precompute()

    # ------------------------------------------------------------------
    def _fixture(self, home: str, away: str) -> dict:
        return {"home": home, "away": away, "neutral": True,
                "tournament": "FIFA World Cup", "date": self.kickoff,
                "country": _venue(home, away)}

    def _precompute(self) -> None:
        # Group-stage scoreline grids (72 matches).
        self.group_matches: list[tuple[str, int, int]] = []
        fixtures = []
        for g, teams in GROUPS.items():
            for a, b in combinations(range(4), 2):
                self.group_matches.append((g, a, b))
                fixtures.append(self._fixture(teams[a], teams[b]))
        probs, lam_h, lam_a = self.predictor.batch_predict(fixtures)
        rho = self.predictor.bundle["rho"]
        self.group_grids = []
        for k in range(len(fixtures)):
            grid = score_grid(lam_h[k], lam_a[k], rho)
            grid = _reconcile(grid, probs[k])
            self.group_grids.append(grid.flatten())
        logger.info("cached %d group-match scoreline grids", len(self.group_grids))

        # Advancement matrix for every ordered knockout pairing.
        n = len(self.teams)
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        fixtures = [self._fixture(self.teams[i], self.teams[j]) for i, j in pairs]
        kprobs, _, _ = self.predictor.batch_predict(fixtures)
        self.P_adv = np.full((n, n), 0.5)
        for (i, j), (pw, pd_, pl) in zip(pairs, kprobs):
            decisive = max(pw + pl, 1e-9)
            self.P_adv[i, j] = pw + pd_ * (pw / decisive)
        logger.info("cached %dx%d knockout advancement matrix", n, n)

    # ------------------------------------------------------------------
    def run(self, n_sims: int = config.N_SIMULATIONS) -> dict:
        rng = self.rng
        n_teams = len(self.teams)

        # ---- sample all group scorelines at once ----------------------
        pts = np.zeros((n_sims, n_teams), dtype=np.int16)
        gf = np.zeros((n_sims, n_teams), dtype=np.int16)
        ga = np.zeros((n_sims, n_teams), dtype=np.int16)
        for (g, a, b), flat in zip(self.group_matches, self.group_grids):
            ta = self.idx[GROUPS[g][a]]
            tb = self.idx[GROUPS[g][b]]
            draws = rng.choice(len(flat), size=n_sims, p=flat / flat.sum())
            hg, agoals = draws // N_GOALS, draws % N_GOALS
            gf[:, ta] += hg; ga[:, ta] += agoals
            gf[:, tb] += agoals; ga[:, tb] += hg
            pts[:, ta] += np.where(hg > agoals, 3, np.where(hg == agoals, 1, 0)).astype(np.int16)
            pts[:, tb] += np.where(agoals > hg, 3, np.where(hg == agoals, 1, 0)).astype(np.int16)

        # ---- group rankings -------------------------------------------
        winners, runners, thirds = {}, {}, {}
        gd = (gf - ga).astype(np.int32)
        for g, teams in GROUPS.items():
            cols = np.array([self.idx[t] for t in teams])
            key = (pts[:, cols].astype(np.float64) * 1e6
                   + gd[:, cols] * 1e3 + gf[:, cols]
                   + rng.random((n_sims, 4)))
            order = np.argsort(-key, axis=1)            # best first
            winners[g] = cols[order[:, 0]]
            runners[g] = cols[order[:, 1]]
            thirds[g] = cols[order[:, 2]]

        # ---- best 8 third-placed teams ---------------------------------
        third_mat = np.stack([thirds[g] for g in GROUP_LETTERS], axis=1)  # (S,12)
        tkey = (np.take_along_axis(pts, third_mat, 1).astype(np.float64) * 1e6
                + np.take_along_axis(gd, third_mat, 1) * 1e3
                + np.take_along_axis(gf, third_mat, 1)
                + rng.random((n_sims, 12)))
        third_order = np.argsort(-tkey, axis=1)[:, :8]   # group indices of best 8
        qual_mask = np.zeros((n_sims, 12), dtype=bool)
        np.put_along_axis(qual_mask, third_order, True, axis=1)
        bitmask = qual_mask.dot(1 << np.arange(12)).astype(np.int32)

        # memoised third-slot allocation per unique qualifying set
        t_slot_specs = [(k, R32[k][2][1]) for k in range(len(R32))
                        if R32[k][2][0] == "T"]
        alloc_cache: dict[int, dict[int, int]] = {}

        def allocation(mask: int) -> dict[int, int]:
            """slot index in R32 -> group-letter index whose third fills it."""
            if mask in alloc_cache:
                return alloc_cache[mask]
            letters = [i for i in range(12) if mask >> i & 1]
            slots = [(k, [GROUP_LETTERS.index(c) for c in spec if GROUP_LETTERS.index(c) in letters])
                     for k, spec in t_slot_specs]
            slots_sorted = sorted(range(len(slots)), key=lambda s: len(slots[s][1]))
            assign: dict[int, int] = {}

            def bt(si: int, used: int) -> bool:
                if si == len(slots_sorted):
                    return True
                k, cand = slots[slots_sorted[si]]
                for letter in cand:
                    if not used >> letter & 1:
                        assign[k] = letter
                        if bt(si + 1, used | 1 << letter):
                            return True
                        del assign[k]
                return False

            if not bt(0, 0):     # extremely unlikely; fall back to greedy order
                free = list(letters)
                for k, cand in slots:
                    pick = next((c for c in cand if c in free), free[0])
                    assign[k] = pick
                    free.remove(pick)
            alloc_cache[mask] = dict(assign)
            return alloc_cache[mask]

        # ---- build R32 lineups ------------------------------------------
        slot_team = np.zeros((n_sims, len(R32), 2), dtype=np.int16)
        for k, (mid, s1, s2) in enumerate(R32):
            for side, spec in ((0, s1), (1, s2)):
                kind = spec[0]
                if kind == "W":
                    slot_team[:, k, side] = winners[spec[1]]
                elif kind == "R":
                    slot_team[:, k, side] = runners[spec[1]]
        # third-place sides, grouped by unique qualifying mask
        for mask in np.unique(bitmask):
            sims = np.nonzero(bitmask == mask)[0]
            for k, letter in allocation(int(mask)).items():
                g = GROUP_LETTERS[letter]
                slot_team[sims, k, 1] = thirds[g][sims]

        # ---- knockout rounds --------------------------------------------
        counts = {r: np.zeros(n_teams, dtype=np.int64)
                  for r in ("round_of_32", "round_of_16", "quarterfinal",
                            "semifinal", "final", "champion")}

        def play(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return np.where(rng.random(len(a)) < self.P_adv[a, b], a, b)

        r32_winner: dict[str, np.ndarray] = {}
        for k, (mid, _, _) in enumerate(R32):
            a, b = slot_team[:, k, 0].astype(int), slot_team[:, k, 1].astype(int)
            np.add.at(counts["round_of_32"], a, 1)
            np.add.at(counts["round_of_32"], b, 1)
            r32_winner[mid] = play(a, b)

        r16_winners = []
        for mid_a, mid_b in R16:
            a, b = r32_winner[mid_a], r32_winner[mid_b]
            np.add.at(counts["round_of_16"], a, 1)
            np.add.at(counts["round_of_16"], b, 1)
            r16_winners.append(play(a, b))

        qf_winners = []
        for i in range(0, 8, 2):
            a, b = r16_winners[i], r16_winners[i + 1]
            np.add.at(counts["quarterfinal"], a, 1)
            np.add.at(counts["quarterfinal"], b, 1)
            qf_winners.append(play(a, b))

        sf_winners = []
        for i in range(0, 4, 2):
            a, b = qf_winners[i], qf_winners[i + 1]
            np.add.at(counts["semifinal"], a, 1)
            np.add.at(counts["semifinal"], b, 1)
            sf_winners.append(play(a, b))

        a, b = sf_winners
        np.add.at(counts["final"], a, 1)
        np.add.at(counts["final"], b, 1)
        champion = play(a, b)
        np.add.at(counts["champion"], champion, 1)

        # ---- aggregate -----------------------------------------------------
        team_group = {t: g for g, ts in GROUPS.items() for t in ts}
        results = []
        for t in self.teams:
            i = self.idx[t]
            results.append({
                "team": t,
                "group": team_group[t],
                "advance_group": counts["round_of_32"][i] / n_sims,
                "round_of_16": counts["round_of_16"][i] / n_sims,
                "quarterfinal": counts["quarterfinal"][i] / n_sims,
                "semifinal": counts["semifinal"][i] / n_sims,
                "final": counts["final"][i] / n_sims,
                "champion": counts["champion"][i] / n_sims,
            })
        results.sort(key=lambda r: r["champion"], reverse=True)
        return {"n_simulations": n_sims,
                "generated_at": datetime.utcnow().isoformat(),
                "teams": results}


def _reconcile(grid: np.ndarray, wdl: np.ndarray) -> np.ndarray:
    """Rescale grid cells so implied W/D/L matches the classifier ensemble."""
    gh, gd_, ga = outcome_probs_from_grid(grid)
    ii, jj = np.meshgrid(range(grid.shape[0]), range(grid.shape[1]), indexing="ij")
    scale = np.ones_like(grid)
    scale[ii > jj] = wdl[0] / max(gh, 1e-9)
    scale[ii == jj] = wdl[1] / max(gd_, 1e-9)
    scale[ii < jj] = wdl[2] / max(ga, 1e-9)
    out = grid * scale
    return out / out.sum()


def run_simulation(n_sims: int = config.N_SIMULATIONS, seed: int = config.RANDOM_SEED,
                   out_path=None) -> dict:
    from prediction.predictor import Predictor
    sim = WorldCupSimulator(Predictor(), seed=seed)
    res = sim.run(n_sims)
    out_path = out_path or (config.REPORTS_DIR / "worldcup2026_simulation.json")
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=2)
    logger.info("wrote %s", out_path)
    return res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=config.N_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    res = run_simulation(args.sims, args.seed)
    print(f"{'team':22s} {'adv%':>6s} {'QF%':>6s} {'SF%':>6s} {'F%':>6s} {'WIN%':>6s}")
    for r in res["teams"][:15]:
        print(f"{r['team']:22s} {100*r['advance_group']:6.1f} {100*r['quarterfinal']:6.1f} "
              f"{100*r['semifinal']:6.1f} {100*r['final']:6.1f} {100*r['champion']:6.1f}")
