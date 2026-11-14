"""Settings are checked locally against the account's OPTIONS /simulations before anything is queued."""

from alphago.brain.options import SimOptions


def test_valid_settings_have_no_problems():
    assert SimOptions().problems({"region": "USA", "delay": 1, "universe": "TOP3000", "neutralization": "INDUSTRY",
                                  "decay": 4, "truncation": 0.08}) == []


def test_each_refusal_is_named():
    probs = SimOptions().problems({"region": "USA", "delay": 1, "universe": "TOP9", "neutralization": "FOO",
                                   "decay": 900, "maxTrade": "ON", "maxPosition": "ON"})
    text = " ".join(probs)
    assert "universe 'TOP9'" in text and "neutralization 'FOO'" in text and "decay 900" in text
    assert "cannot both be ON" in text
