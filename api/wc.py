"""World Cup 2026 team metadata and group schedule for the UI.

Provides the data behind the rich matchup card (flag, code, Elo, rank, group,
host, title odds, recent form) and the group-stage schedule with the model's
favourite per match. Real fixture dates/venues/kickoff times and live scores
are NOT part of this project's data, so they are intentionally absent.
"""
from __future__ import annotations

import json
from itertools import combinations

import config
from simulation.worldcup2026 import GROUPS, HOSTS

# name -> (FIFA 3-letter code, ISO-3166 alpha-2 OR explicit flag emoji)
WC_TEAMS: dict[str, tuple[str, str]] = {
    "Mexico": ("MEX", "MX"), "South Africa": ("RSA", "ZA"),
    "South Korea": ("KOR", "KR"), "Czech Republic": ("CZE", "CZ"),
    "Canada": ("CAN", "CA"), "Bosnia and Herzegovina": ("BIH", "BA"),
    "Qatar": ("QAT", "QA"), "Switzerland": ("SUI", "CH"),
    "Brazil": ("BRA", "BR"), "Morocco": ("MAR", "MA"),
    "Haiti": ("HAI", "HT"), "Scotland": ("SCO", "🏴\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f"),
    "United States": ("USA", "US"), "Paraguay": ("PAR", "PY"),
    "Australia": ("AUS", "AU"), "Turkey": ("TUR", "TR"),
    "Germany": ("GER", "DE"), "Curaçao": ("CUW", "CW"),
    "Ivory Coast": ("CIV", "CI"), "Ecuador": ("ECU", "EC"),
    "Netherlands": ("NED", "NL"), "Japan": ("JPN", "JP"),
    "Sweden": ("SWE", "SE"), "Tunisia": ("TUN", "TN"),
    "Belgium": ("BEL", "BE"), "Egypt": ("EGY", "EG"),
    "Iran": ("IRN", "IR"), "New Zealand": ("NZL", "NZ"),
    "Spain": ("ESP", "ES"), "Cape Verde": ("CPV", "CV"),
    "Saudi Arabia": ("KSA", "SA"), "Uruguay": ("URU", "UY"),
    "France": ("FRA", "FR"), "Senegal": ("SEN", "SN"),
    "Iraq": ("IRQ", "IQ"), "Norway": ("NOR", "NO"),
    "Argentina": ("ARG", "AR"), "Algeria": ("ALG", "DZ"),
    "Austria": ("AUT", "AT"), "Jordan": ("JOR", "JO"),
    "Portugal": ("POR", "PT"), "DR Congo": ("COD", "CD"),
    "Uzbekistan": ("UZB", "UZ"), "Colombia": ("COL", "CO"),
    "England": ("ENG", "🏴\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f"),
    "Croatia": ("CRO", "HR"), "Ghana": ("GHA", "GH"), "Panama": ("PAN", "PA"),
}

_GROUP_OF = {t: g for g, teams in GROUPS.items() for t in teams}


def _flag(iso_or_emoji: str) -> str:
    if len(iso_or_emoji) == 2 and iso_or_emoji.isalpha():
        return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in iso_or_emoji.upper())
    return iso_or_emoji  # already an emoji (Scotland / England)


def _recent_form(predictor, team: str, n: int = 5) -> list[str]:
    st = predictor.builder.teams.get(team)
    if not st or not st.history:
        return []
    out = []
    for m in list(st.history)[-n:]:
        out.append("W" if m.gf > m.ga else "D" if m.gf == m.ga else "L")
    return out


def _title_odds() -> dict[str, float]:
    path = config.REPORTS_DIR / "worldcup2026_simulation.json"
    if not path.exists():
        return {}
    teams = json.loads(path.read_text()).get("teams", [])
    return {t["team"]: t.get("champion", 0.0) for t in teams}


def team_meta(predictor) -> dict[str, dict]:
    """Per-team card data for all 48 World Cup teams."""
    ranks = {r["team"]: r for r in predictor.elo_table(top=10_000)}
    titles = _title_odds()
    meta = {}
    for name, (code, flagspec) in WC_TEAMS.items():
        r = ranks.get(name, {})
        meta[name] = {
            "code": code,
            "flag": _flag(flagspec),
            "group": _GROUP_OF.get(name),
            "host": name in HOSTS,
            "elo": r.get("elo"),
            "rank": r.get("rank"),
            "title": titles.get(name, 0.0),
            "form": _recent_form(predictor, name),
        }
    return meta


def group_schedule(predictor) -> list[dict]:
    """All 72 group-stage matches with the model's favourite. Hosts play at
    home; everyone else is treated as neutral."""
    out = []
    for g, teams in GROUPS.items():
        for a, b in combinations(teams, 2):
            home, away = a, b
            if away in HOSTS and home not in HOSTS:
                home, away = away, home
            neutral = home not in HOSTS
            hw, draw, aw = predictor.win_draw_loss(home, away, neutral=neutral)
            if hw >= aw:
                fav, fav_p = home, hw
            else:
                fav, fav_p = away, aw
            out.append({
                "group": g,
                "home": home, "away": away,
                "home_code": WC_TEAMS[home][0], "away_code": WC_TEAMS[away][0],
                "home_flag": _flag(WC_TEAMS[home][1]), "away_flag": _flag(WC_TEAMS[away][1]),
                "fav_code": WC_TEAMS[fav][0], "fav_prob": round(fav_p, 3),
                "neutral": neutral,
            })
    return out
