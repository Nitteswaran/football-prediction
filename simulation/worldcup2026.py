"""2026 FIFA World Cup structure (official draw of 5 December 2025).

Team names use the spelling of the historical results dataset.
"""
from __future__ import annotations

GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

HOSTS = {"Mexico": "Mexico", "Canada": "Canada", "United States": "United States"}

# Round of 32 (matches 73..88). Slots:
#   ("W", g)  winner of group g | ("R", g) runner-up | ("T", frozenset) best third
# from one of the listed groups.
R32: list[tuple] = [
    ("M74", ("W", "E"), ("T", "ABCDF")),
    ("M77", ("W", "I"), ("T", "CDFGH")),
    ("M73", ("R", "A"), ("R", "B")),
    ("M75", ("W", "F"), ("R", "C")),
    ("M76", ("W", "C"), ("R", "F")),
    ("M78", ("R", "E"), ("R", "I")),
    ("M79", ("W", "A"), ("T", "CEFHI")),
    ("M80", ("W", "L"), ("T", "EHIJK")),
    ("M83", ("R", "K"), ("R", "L")),
    ("M84", ("W", "H"), ("R", "J")),
    ("M81", ("W", "D"), ("T", "BEFIJ")),
    ("M82", ("W", "G"), ("T", "AEHIJ")),
    ("M86", ("W", "J"), ("R", "H")),
    ("M88", ("R", "D"), ("R", "G")),
    ("M85", ("W", "B"), ("T", "EFGIJ")),
    ("M87", ("W", "K"), ("T", "DEIJL")),
]

# Round of 16 pairings by R32 match id (official matches 89..96),
# ordered so that successive pairs feed QFs, then SFs, then the final.
R16: list[tuple[str, str]] = [
    ("M74", "M77"),   # M89
    ("M73", "M75"),   # M90
    ("M76", "M78"),   # M91
    ("M79", "M80"),   # M92
    ("M83", "M84"),   # M93
    ("M81", "M82"),   # M94
    ("M86", "M88"),   # M95
    ("M85", "M87"),   # M96
]
# QF: (89 v 90), (91 v 92), (93 v 94), (95 v 96); SF: (QF1 v QF2), (QF3 v QF4).

ROUND_NAMES = ["group_stage", "round_of_32", "round_of_16", "quarterfinal",
               "semifinal", "final", "champion"]
