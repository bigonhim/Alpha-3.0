"""BRAIN's sentence for each check, as alpha-harness messages.ts / brain.ts word and format it."""

from __future__ import annotations

import pytest

from alphago.ui.check_messages import (
    CHECK_MESSAGES,
    brain_value,
    by_result,
    cell,
    check_html,
    check_message,
    check_parts,
)


def test_every_template_is_keyed_by_stage_check_and_result() -> None:
    assert len(CHECK_MESSAGES) >= 300
    assert all(k.startswith(("IS.", "OS.")) for k in CHECK_MESSAGES)
    assert CHECK_MESSAGES["IS.LOW_SHARPE.FAIL"] == "Sharpe of **{value}** is below cutoff of **{limit}**."
    assert "IS.CONCENTRATED_WEIGHT.WITH_VALUES.FAIL" in CHECK_MESSAGES
    assert "IS.LOW_ROBUST_UNIVERSE_SHARPE.WITH_RATIO.FAIL" in CHECK_MESSAGES
    assert "IS.MATCHES_COMPETITION.PASS.multiple.beginning" in CHECK_MESSAGES


@pytest.mark.parametrize("value,kind,in_check,expected", [
    (1.234, "decimal", False, "1.23"),
    (1.0, "decimal", True, "1"),
    (1.0, "decimal", False, "1.00"),
    (0.25, "percent", True, "25%"),
    (0.2534, "percent", True, "25.34%"),
    (0.0007, "permyriad", False, "7.00 bps"),
    (12345.678, "decimal", False, "12,345.68"),
    (-0.001, "decimal", False, "0.00"),
])
def test_brain_value(value, kind, in_check, expected) -> None:
    assert brain_value(value, kind, in_check) == expected


def test_cell_blank_counts_and_labels() -> None:
    assert cell(None, "decimal") == ""
    assert cell(1400, "integer") == "1400"
    assert cell("TRAIN", "decimal") == "TRAIN"
    assert cell(0.1, "percent") == "10.00%"


def test_decimal_and_percent_checks() -> None:
    assert check_message({"name": "LOW_SHARPE", "result": "FAIL", "value": 1.234, "limit": 1.25}) == \
        "Sharpe of **1.23** is below cutoff of **1.25**."
    assert check_message({"name": "HIGH_TURNOVER", "result": "PASS", "value": 0.25, "limit": 0.7}) == \
        "Turnover of **25%** is below cutoff of **70%**."
    assert check_message({"name": "LOW_FITNESS", "result": "PASS", "value": 1.5, "limit": 1}, markdown=False) == \
        "Fitness of 1.50 is above cutoff of 1."


def test_null_value_uses_the_null_wording() -> None:
    text = check_message({"name": "LOW_2Y_SHARPE", "result": "PASS", "value": None, "limit": 1.58})
    assert text == CHECK_MESSAGES["IS.LOW_2Y_SHARPE.PASS.null"].replace("{limit}", "1.58")


def test_unknown_check_falls_back_to_name_and_result() -> None:
    assert check_message({"name": "SOMETHING_NEW", "result": "WARNING"}) == "SOMETHING_NEW WARNING"
    assert check_message({"name": "SOMETHING_NEW"}) == "SOMETHING_NEW PENDING"


def test_os_stage_has_its_own_wording() -> None:
    assert check_message({"name": "SHARPE", "result": "PASS"}, "OS") == "Sharpe check passed."


def test_robust_universe_with_ratio() -> None:
    text = check_message({"name": "LOW_ROBUST_UNIVERSE_SHARPE", "result": "FAIL", "value": 0.5, "limit": 1.0,
                          "ratio": 0.4})
    assert "**0.50**" in text and "**1**" in text and "40%" in text


def test_concentrated_weight_with_values_and_date() -> None:
    text = check_message({"name": "CONCENTRATED_WEIGHT", "result": "FAIL", "value": 0.12, "limit": 0.1,
                          "date": "2020-01-05"})
    assert text == "Weight concentration **12%** is above cutoff of **10%** on **1/5/2020**."
    plain = check_message({"name": "CONCENTRATED_WEIGHT", "result": "PASS"})
    assert plain == "Weight is well distributed over instruments."


def test_competitions_are_links() -> None:
    one = check_parts({"name": "MATCHES_COMPETITION", "result": "PASS", "competitions": [{"id": "IQC26", "name": "IQC"}]})
    assert [p.text for p in one] == ["Competition ", "IQC", " matches."]
    assert one[1].href == "https://platform.worldquantbrain.com/competitions/IQC26"
    many = check_message({"name": "MATCHES_COMPETITION", "result": "PASS",
                          "competitions": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}]})
    assert many == ("These competitions match: [A](https://platform.worldquantbrain.com/competitions/a), "
                    "[B](https://platform.worldquantbrain.com/competitions/b).")
    assert '<a href="https://platform.worldquantbrain.com/competitions/a"' in check_html(
        {"name": "MATCHES_COMPETITION", "result": "PASS", "competitions": [{"id": "a", "name": "A"}]})


def test_themes_with_multipliers() -> None:
    single = check_message({"name": "MATCHES_THEMES", "result": "PASS", "themes": [{"name": "Value", "multiplier": 2}]})
    assert single == "Theme **Value** matches with multiplier of **2**."
    many = check_message({"name": "MATCHES_THEMES", "result": "PASS", "multiplier": 3,
                          "themes": [{"name": "A", "multiplier": 2}, {"name": "B", "multiplier": 1.5}]})
    assert "**A — 2; B — 1.5**" in many and "**3**" in many


def test_pyramids_and_classifications() -> None:
    pyr = check_message({"name": "MATCHES_PYRAMID", "result": "PASS", "effective": 4,
                         "pyramids": [{"name": "USA/D1/PV", "multiplier": 1.2}]})
    assert "**USA/D1/PV**" in pyr and "**1.2**" in pyr and "**4**" in pyr
    cls = check_message({"name": "MATCHES_CLASSIFICATION", "result": "PASS",
                         "matched": [{"name": "Power Pool Alpha"}, {"name": "Regular Alpha"}]})
    assert cls == "Classifications **Power Pool Alpha; Regular Alpha** match"


def test_data_diversity_not_passed_and_authorization_lists() -> None:
    assert check_message({"name": "DATA_DIVERSITY", "result": "WARNING", "dataCategoryName": "Fundamental",
                          "fields": ["a", "b"]}) == 'Overused data **"a", "b"** field(s) in **Fundamental**.'
    assert check_message({"name": "DATA_SET_AUTHORIZATION", "result": "FAIL", "datasets": ["x", "y"]}) == \
        "Invalid datasets: x, y"
    assert check_message({"name": "OPERATOR_AUTHORIZATION", "result": "FAIL", "operators": "ts_foo"}) == \
        "Invalid operators: ts_foo"


def test_missing_value_reads_none_where_the_template_uses_it() -> None:
    text = check_message({"name": "SELF_CORRELATION", "result": "PENDING"})
    assert text == "Self-correlation check pending."


def test_ladder_dates_are_days_in_utc() -> None:
    text = check_message({"name": "IS_LADDER_SHARPE", "result": "FAIL", "value": 1.1, "limit": 1.5, "year": 2,
                          "startDate": "2019-01-01", "endDate": "2020-12-31"})
    assert "**1/1/2019**…**12/31/2020**" in text


def test_html_escapes_brain_text() -> None:
    out = check_html({"name": "COMPILE_ERROR", "result": "FAIL", "message": "<b>bad</b>"})
    assert "&lt;b&gt;" in out and "<b>" not in out


def test_grouped_in_brains_order() -> None:
    groups = by_result([{"name": "A", "result": "PENDING"}, {"name": "B", "result": "FAIL"},
                        {"name": "C", "result": "PASS"}, {"name": "D"}, {"name": "E", "result": "ODD"}])
    assert [g for g, _ in groups] == ["PASS", "FAIL", "PENDING", "ODD"]
    assert [c["name"] for c in dict(groups)["PENDING"]] == ["A", "D"]
