"""BRAIN's own sentence for every check result, as its Alpha page words it.

Ported from alpha-harness ``frontend/src/screens/alpha/messages.ts`` (the templates, verbatim) and ``brain.ts``
(``checkMessage``, ``message``, ``brainValue``): a consultant reading a check here reads exactly what BRAIN's
website would say, figures formatted the way BRAIN formats them (Sharpe and Fitness as decimals, Turnover, Returns,
Imbalance and Weight as percents, a whole number without decimals).

Templates are keyed ``IS.<CHECK>.<RESULT>`` or ``OS.<CHECK>.<RESULT>``; ``.null`` is the wording BRAIN uses when the
check's value is null, ``.WITH_VALUES`` / ``.WITH_RATIO`` the wording when the check carries those figures, and
``.single`` / ``.multiple`` (with ``.beginning`` / ``.end`` around competition links) the wording for matches.
``**x**`` is a bold run; ``{name}`` is filled from the check. An unknown check reads ``{NAME} {RESULT}``.
"""

from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

PLATFORM_COMPETITIONS_URL = "https://platform.worldquantbrain.com/competitions/"

CHECK_MESSAGES: dict[str, str] = {
    'IS.ALPHA_AIAC_INELIGIBILITY_TAGS.WARNING': 'Alpha description/tag is not matching requirements of AIACv2.',
    'IS.ALPHA_DECOMMISSIONED.FAIL': 'Alpha is decommissioned',
    'IS.ALREADY_SUBMITTED.FAIL': 'Alpha already submitted.',
    'IS.AVAILABLE_SETTING.FAIL': 'Alpha settings are unsubmittable.',
    'IS.CLUSTER_TEST.ERROR': 'Cluster Test check error.',
    'IS.CLUSTER_TEST.PASS':
        'Cluster Test Sharpe **{value}** is above cutoff of **{limit}**. Alpha qualifies as Cluster Alpha.',
    'IS.CLUSTER_TEST.WARNING':
        ('Cluster Test Sharpe **{value}** is below cutoff of **{limit}**. Alpha does not qualify as Cluster '
         'Alpha.'),
    'IS.COMBO_DESCRIPTION_LENGTH.FAIL': 'Combo description length **{value}** is below cutoff of **{limit}**.',
    'IS.COMPILE_ERROR.FAIL': '{message}',
    'IS.COMPONENT_ALPHA_AUTHORIZATION.ERROR': 'Unauthorized component alphas check error.',
    'IS.COMPONENT_ALPHA_AUTHORIZATION.FAIL':
        'Your SuperAlpha contains unauthorized component alphas, please clone this alpha and re-simulate.',
    'IS.CONCENTRATED_WEIGHT.ERROR': 'Weight check error.',
    'IS.CONCENTRATED_WEIGHT.FAIL': 'Weight is too strongly concentrated or too few instruments are assigned weight.',
    'IS.CONCENTRATED_WEIGHT.PASS': 'Weight is well distributed over instruments.',
    'IS.CONCENTRATED_WEIGHT.PENDING': 'Weight concentration test pending.',
    'IS.CONCENTRATED_WEIGHT.WARNING': 'Weight is too strongly concentrated or too few instruments are assigned weight.',
    'IS.CONCENTRATED_WEIGHT.WITH_VALUES.FAIL':
        'Weight concentration **{value}** is above cutoff of **{limit}** on **{date}**.',
    'IS.CONCENTRATED_WEIGHT.WITH_VALUES.PASS':
        'Weight is well distributed over instruments. **{value}** is below cutoff of **{limit}**',
    'IS.CONCENTRATED_WEIGHT.WITH_VALUES.WARNING':
        'Weight concentration **{value}** is above cutoff of **{limit}** on **{date}**.',
    'IS.D0_SUBMISSION.FAIL': 'D0 submissions **{value}** reached quota of **{limit}**.',
    'IS.D0_SUBMISSION.PASS': 'D0 submissions **{value}** below quota of **{limit}**.',
    'IS.D0_SUBMISSION.PENDING': 'D0 submissions quota check pending.',
    'IS.D0_SUBMISSION.WARNING': 'D0 submissions **{value}** approaching quota of **{limit}**.',
    'IS.DATA_DIVERSITY.NOT_PASSED': 'Overused data **{field}** field(s) in **{dataCategoryName}**.',
    'IS.DATA_DIVERSITY.PASS': 'Data overuse check passed.',
    'IS.DATA_DIVERSITY.PENDING': 'Data overuse check pending.',
    'IS.DATA_SET_AUTHORIZATION.FAIL': 'Invalid datasets: {datasets}',
    'IS.GLOBAL_SUMMIT_SUBMISSION.FAIL': 'Submissions disabled till 23:59 PM EST, 8 Dec 2023',
    'IS.HIGH_TURNOVER.ERROR': 'Turnover check error.',
    'IS.HIGH_TURNOVER.FAIL': 'Turnover of **{value}** is above cutoff of **{limit}**.',
    'IS.HIGH_TURNOVER.PASS': 'Turnover of **{value}** is below cutoff of **{limit}**.',
    'IS.HT_AFTER_COST_SHARPE.PASS':
        'After Cost High Turnover: After cost Sharpe of **{value}** is above cutoff of **{limit}**.',
    'IS.HT_AFTER_COST_SHARPE.WARNING':
        'After Cost High Turnover: After cost Sharpe of **{value}** is below cutoff of **{limit}**.',
    'IS.HT_HIGH_TURNOVER_RETURNS_RATIO.PASS':
        'High Turnover: High Turnover returns ratio of **{value}** is above cutoff of **{limit}**.',
    'IS.HT_HIGH_TURNOVER_RETURNS_RATIO.WARNING':
        'High Turnover: High Turnover returns ratio of **{value}** is below cutoff of **{limit}**.',
    'IS.HT_INVESTABLE_MAX_POSITION_SHARPE.PASS':
        'Investable High Turnover: Max Position Sharpe of **{value}** is above cutoff of **{limit}**.',
    'IS.HT_INVESTABLE_MAX_POSITION_SHARPE.WARNING':
        'Investable High Turnover: Max Position Sharpe of **{value}** is below cutoff of **{limit}**.',
    'IS.HT_INVESTABLE_MAX_POSITION_TURNOVER.PASS':
        'Investable High Turnover: Max Position turnover of **{value}** is above cutoff of **{limit}**.',
    'IS.HT_INVESTABLE_MAX_POSITION_TURNOVER.WARNING':
        'Investable High Turnover: Max Position turnover of **{value}** is below cutoff of **{limit}**.',
    'IS.HT_INVESTABLE_MAX_TRADE_SHARPE.PASS':
        'Investable High Turnover: Max Trade Sharpe of **{value}** is above cutoff of **{limit}**.',
    'IS.HT_INVESTABLE_MAX_TRADE_SHARPE.WARNING':
        'Investable High Turnover: Max Trade Sharpe of **{value}** is below cutoff of **{limit}**.',
    'IS.HT_INVESTABLE_MAX_TRADE_TURNOVER.PASS':
        'Investable High Turnover: Max Trade turnover of **{value}** is above cutoff of **{limit}**.',
    'IS.HT_INVESTABLE_MAX_TRADE_TURNOVER.WARNING':
        'Investable High Turnover: Max Trade turnover of **{value}** is below cutoff of **{limit}**.',
    'IS.HT_LIQUID_MINVOL10M_TOPDIV3000_SHARPE_RATIO.PASS':
        ('Liquid High Turnover: MINVOL10M and TOPDIV3000 Sharpe ratio of **{value}** is above cutoff of '
         '**{limit}**.'),
    'IS.HT_LIQUID_MINVOL10M_TOPDIV3000_SHARPE_RATIO.WARNING':
        ('Liquid High Turnover: MINVOL10M and TOPTOPDIV3000200 Sharpe ratio of **{value}** is below cutoff of '
         '**{limit}**.'),
    'IS.HT_LIQUID_TOP200_SHARPE.PASS':
        'Liquid High Turnover: TOP200 Sharpe of **{value}** is above cutoff of **{limit}**.',
    'IS.HT_LIQUID_TOP200_SHARPE.WARNING':
        'Liquid High Turnover: TOP200 Sharpe of **{value}** is below cutoff of **{limit}**.',
    'IS.HT_LIQUID_TOP500_TOP200_SHARPE_RATIO.PASS':
        'Liquid High Turnover: TOP500 and TOP200 Sharpe ratio of **{value}** is above cutoff of **{limit}**.',
    'IS.HT_LIQUID_TOP500_TOP200_SHARPE_RATIO.WARNING':
        'Liquid High Turnover: TOP500 and TOP200 Sharpe ratio of **{value}** is below cutoff of **{limit}**.',
    'IS.HT_LIQUID_TOPDIV3000_SHARPE.PASS':
        'Liquid High Turnover: TOPDIV3000 Sharpe of **{value}** is above cutoff of **{limit}**.',
    'IS.HT_LIQUID_TOPDIV3000_SHARPE.WARNING':
        'Liquid High Turnover: TOPDIV3000 Sharpe of **{value}** is below cutoff of **{limit}**.',
    'IS.HT_ORTHOGONAL_RAM_NEUTRALIZATION.PASS':
        ('Orthogonal High Turnover: Neutralization of **{value}** matches Orthogonal High Turnover '
         'neutralization of **{limit}**.'),
    'IS.HT_ORTHOGONAL_RAM_NEUTRALIZATION.WARNING':
        ('Orthogonal High Turnover: Neutralization of **{value}** does not match Orthogonal High Turnover '
         'neutralization of **{limit}**.'),
    'IS.HT_PNL_REALIZATION_HORIZON.PASS':
        'High Turnover: Pnl realization of **{value}** is below cutoff of **{limit}**.',
    'IS.HT_PNL_REALIZATION_HORIZON.WARNING':
        'High Turnover: Pnl realization of **{value}** is above cutoff of **{limit}**.',
    'IS.HT_TURNOVER.PASS': 'High Turnover: Turnover of **{value}** is above cutoff of **{limit}**.',
    'IS.HT_TURNOVER.WARNING': 'High Turnover: Turnover of **{value}** is below cutoff of **{limit}**.',
    'IS.IMBALANCE.ERROR': 'Imbalance check error.',
    'IS.IMBALANCE.FAIL':
        'Imbalance of **{value}** is above cutoff of **{limit}** on **{date}**. Check neutralization setting.',
    'IS.IMBALANCE.PASS': 'Imbalance of **{value}** is below cutoff of **{limit}**.',
    'IS.IMBALANCE.PENDING': 'Imbalance check pending.',
    'IS.IQC2026S3_NON_THEME_SUBMISSION.FAIL': 'Non-theme submissions **{value}** reached quota of **{limit}**.',
    'IS.IQC2026S3_NON_THEME_SUBMISSION.PASS': 'Non-theme submissions **{value}** below quota of **{limit}**.',
    'IS.IQC2026S3_NON_THEME_SUBMISSION.WARNING': 'Non-theme submissions **{value}** approaching quota of **{limit}**.',
    'IS.IS_LADDER_SHARPE.ERROR': 'IS ladder Sharpe check error.',
    'IS.IS_LADDER_SHARPE.FAIL':
        ('IS ladder Sharpe of **{value}** is below cutoff of **{limit}** for ladder year **{year}**: '
         '**{startDate}**…**{endDate}**.'),
    'IS.IS_LADDER_SHARPE.PASS':
        ('IS ladder Sharpe of **{value}** is above cutoff of **{limit}** for ladder year **{year}**: '
         '**{startDate}**…**{endDate}**.'),
    'IS.IS_LADDER_SHARPE.PENDING': 'IS ladder Sharpe check pending.',
    'IS.IS_LADDER_SHARPE.WARNING':
        ('IS ladder Sharpe of **{value}** is below cutoff of **{limit}** for ladder year **{year}**: '
         '**{startDate}**…**{endDate}**.'),
    'IS.LONG_DURATION.FAIL': 'The simulation duration is too short please clone this alpha and re-simulate.',
    'IS.LOW_2Y_SHARPE.ERROR': '2 year Sharpe check error.',
    'IS.LOW_2Y_SHARPE.FAIL': '2 year Sharpe of **{value}** is below cutoff of **{limit}**.',
    'IS.LOW_2Y_SHARPE.PASS': '2 year Sharpe of **{value}** is above cutoff of **{limit}**.',
    'IS.LOW_2Y_SHARPE.PASS.null': '2 year Sharpe is above cutoff of **{limit}**.',
    'IS.LOW_2Y_SHARPE.WARNING': '2 year Sharpe of **{value}** is below cutoff of **{limit}**.',
    'IS.LOW_AFTER_COST_ILLIQUID_UNIVERSE_SHARPE.ERROR': 'Most illiquid 50% instruments after cost Sharpe check error.',
    'IS.LOW_AFTER_COST_ILLIQUID_UNIVERSE_SHARPE.FAIL':
        ('Most illiquid 50% instruments after cost Sharpe of **{value}** is below cutoff of **{limit}** '
         '({afterCostSharpe} original universe after cost Sharpe).'),
    'IS.LOW_AFTER_COST_ILLIQUID_UNIVERSE_SHARPE.PASS':
        ('Most illiquid 50% instruments after cost Sharpe of **{value}** is above cutoff of **{limit}** '
         '({afterCostSharpe} original universe after cost Sharpe).'),
    'IS.LOW_AFTER_COST_ILLIQUID_UNIVERSE_SHARPE.WARNING':
        ('Most illiquid 50% instruments after cost Sharpe of **{value}** is below cutoff of **{limit}** '
         '({afterCostSharpe} original universe after cost Sharpe).'),
    'IS.LOW_ASI_JPN_SHARPE.ERROR': 'Japan Robustness Sharpe check error.',
    'IS.LOW_ASI_JPN_SHARPE.FAIL': 'Japan Robustness Sharpe of **{value}** is below cutoff of **{limit}**.',
    'IS.LOW_ASI_JPN_SHARPE.PASS': 'Japan Robustness Sharpe of **{value}** is above cutoff of **{limit}**.',
    'IS.LOW_ASI_JPN_SHARPE.WARNING': 'Japan Robustness Sharpe of **{value}** is below cutoff of **{limit}**.',
    'IS.LOW_COVERAGE.ERROR': 'Zero coverage length check error.',
    'IS.LOW_COVERAGE.FAIL': 'Zero coverage length of **{value}** days is above cutoff of **{limit}** days.',
    'IS.LOW_DURATION.ERROR': 'Simulation length check error.',
    'IS.LOW_DURATION.FAIL': 'Simulation length of **{value}** days is below cutoff of **{limit}** days.',
    'IS.LOW_FITNESS.ERROR': 'Fitness check error.',
    'IS.LOW_FITNESS.FAIL': 'Fitness of **{value}** is below cutoff of **{limit}**.',
    'IS.LOW_FITNESS.PASS': 'Fitness of **{value}** is above cutoff of **{limit}**.',
    'IS.LOW_FITNESS.WARNING': 'Fitness of **{value}** is below cutoff of **{limit}**.',
    'IS.LOW_GLB_AMER_SHARPE.ERROR': 'AMER Sharpe check error.',
    'IS.LOW_GLB_AMER_SHARPE.FAIL': 'AMER Sharpe of **{value}** is below cutoff of **{limit}**.',
    'IS.LOW_GLB_AMER_SHARPE.PASS': 'AMER Sharpe of **{value}** is above cutoff of **{limit}**.',
    'IS.LOW_GLB_AMER_SHARPE.WARNING': 'AMER Sharpe of **{value}** is below cutoff of **{limit}**.',
    'IS.LOW_GLB_APAC_SHARPE.ERROR': 'APAC Sharpe check error.',
    'IS.LOW_GLB_APAC_SHARPE.FAIL': 'APAC Sharpe of **{value}** is below cutoff of **{limit}**.',
    'IS.LOW_GLB_APAC_SHARPE.PASS': 'APAC Sharpe of **{value}** is above cutoff of **{limit}**.',
    'IS.LOW_GLB_APAC_SHARPE.WARNING': 'APAC Sharpe of **{value}** is below cutoff of **{limit}**.',
    'IS.LOW_GLB_EMEA_SHARPE.ERROR': 'EMEA Sharpe check error.',
    'IS.LOW_GLB_EMEA_SHARPE.FAIL': 'EMEA Sharpe of **{value}** is below cutoff of **{limit}**.',
    'IS.LOW_GLB_EMEA_SHARPE.PASS': 'EMEA Sharpe of **{value}** is above cutoff of **{limit}**.',
    'IS.LOW_GLB_EMEA_SHARPE.WARNING': 'EMEA Sharpe of **{value}** is below cutoff of **{limit}**.',
    'IS.LOW_INVESTABILITY_CONSTRAINED_SHARPE.ERROR': 'Investability Constrained Sharpe check error.',
    'IS.LOW_INVESTABILITY_CONSTRAINED_SHARPE.FAIL':
        'Investability Constrained Sharpe of **{value}** is below cutoff of **{limit}**.',
    'IS.LOW_INVESTABILITY_CONSTRAINED_SHARPE.PASS':
        'Investability Constrained Sharpe of **{value}** is above cutoff of **{limit}**.',
    'IS.LOW_INVESTABILITY_CONSTRAINED_SHARPE.WARNING':
        'Investability Constrained Sharpe of **{value}** is below cutoff of **{limit}**.',
    'IS.LOW_RETURNS.ERROR': 'Returns check error.',
    'IS.LOW_RETURNS.FAIL': 'Returns of **{value}** is below cutoff of **{limit}**.',
    'IS.LOW_RETURNS.PASS': 'Returns of **{value}** is above cutoff of **{limit}**.',
    'IS.LOW_RETURNS.WARNING': 'Returns of **{value}** is below cutoff of **{limit}**.',
    'IS.LOW_ROBUST_UNIVERSE_RETURNS.ERROR': 'Robust universe returns check error.',
    'IS.LOW_ROBUST_UNIVERSE_RETURNS.FAIL':
        'Robust universe returns of **{value}** is below cutoff of **{limit}** ({ratio} of Alpha).',
    'IS.LOW_ROBUST_UNIVERSE_RETURNS.PASS':
        'Robust universe returns of **{value}** is above cutoff of **{limit}** ({ratio} of Alpha).',
    'IS.LOW_ROBUST_UNIVERSE_RETURNS.WARNING':
        'Robust universe returns of **{value}** is below cutoff of **{limit}** ({ratio} of Alpha).',
    'IS.LOW_ROBUST_UNIVERSE_SHARPE.ERROR': 'Robust universe Sharpe check error.',
    'IS.LOW_ROBUST_UNIVERSE_SHARPE.FAIL': 'Robust universe Sharpe of **{value}** is below cutoff of **{limit}**.',
    'IS.LOW_ROBUST_UNIVERSE_SHARPE.PASS': 'Robust universe Sharpe of **{value}** is above cutoff of **{limit}**.',
    'IS.LOW_ROBUST_UNIVERSE_SHARPE.WARNING': 'Robust universe Sharpe of **{value}** is below cutoff of **{limit}**.',
    'IS.LOW_ROBUST_UNIVERSE_SHARPE.WITH_RATIO.ERROR': 'Robust universe Sharpe check error.',
    'IS.LOW_ROBUST_UNIVERSE_SHARPE.WITH_RATIO.FAIL':
        'Robust universe Sharpe of **{value}** is below cutoff of **{limit}** ({ratio} of Alpha).',
    'IS.LOW_ROBUST_UNIVERSE_SHARPE.WITH_RATIO.PASS':
        'Robust universe Sharpe of **{value}** is above cutoff of **{limit}** ({ratio} of Alpha).',
    'IS.LOW_ROBUST_UNIVERSE_SHARPE.WITH_RATIO.WARNING':
        'Robust universe Sharpe of **{value}** is below cutoff of **{limit}** ({ratio} of Alpha).',
    'IS.LOW_SHARPE.ERROR': 'Sharpe check error.',
    'IS.LOW_SHARPE.FAIL': 'Sharpe of **{value}** is below cutoff of **{limit}**.',
    'IS.LOW_SHARPE.PASS': 'Sharpe of **{value}** is above cutoff of **{limit}**.',
    'IS.LOW_SHARPE.WARNING': 'Sharpe of **{value}** is below cutoff of **{limit}**.',
    'IS.LOW_SUB_UNIVERSE_SHARPE.ERROR': 'Sub-universe universe Sharpe check error.',
    'IS.LOW_SUB_UNIVERSE_SHARPE.FAIL': 'Sub-universe Sharpe of **{value}** is below cutoff of **{limit}**.',
    'IS.LOW_SUB_UNIVERSE_SHARPE.PASS': 'Sub-universe Sharpe of **{value}** is above cutoff of **{limit}**.',
    'IS.LOW_SUB_UNIVERSE_SHARPE.WARNING':
        'There is a high likelihood of failing the sub-universe check after submission.',
    'IS.LOW_TURNOVER.ERROR': 'Turnover check error.',
    'IS.LOW_TURNOVER.FAIL': 'Turnover of **{value}** is below cutoff of **{limit}**.',
    'IS.LOW_TURNOVER.PASS': 'Turnover of **{value}** is above cutoff of **{limit}**.',
    'IS.MATCHES_CLASSIFICATION.PASS.multiple': 'Classifications **{value}** match',
    'IS.MATCHES_CLASSIFICATION.PASS.single': 'Classification **{value}** matches',
    'IS.MATCHES_COMPETITION.ERROR': 'Competition check error.',
    'IS.MATCHES_COMPETITION.PASS.multiple': 'These competitions match: **{matches}**.',
    'IS.MATCHES_COMPETITION.PASS.multiple.beginning': 'These competitions match: ',
    'IS.MATCHES_COMPETITION.PASS.multiple.end': '.',
    'IS.MATCHES_COMPETITION.PASS.single': 'Competition **{name}** matches.',
    'IS.MATCHES_COMPETITION.PASS.single.beginning': 'Competition ',
    'IS.MATCHES_COMPETITION.PASS.single.end': ' matches.',
    'IS.MATCHES_COMPETITION.PENDING': 'Competition check pending.',
    'IS.MATCHES_COMPETITION.PENDING.multiple': 'Competition check pending: **{matches}**.',
    'IS.MATCHES_COMPETITION.PENDING.multiple.beginning': 'Competition check pending: ',
    'IS.MATCHES_COMPETITION.PENDING.multiple.end': '.',
    'IS.MATCHES_COMPETITION.PENDING.single': 'Competition check pending: **{name}**.',
    'IS.MATCHES_COMPETITION.PENDING.single.beginning': 'Competition check pending: ',
    'IS.MATCHES_COMPETITION.PENDING.single.end': '.',
    'IS.MATCHES_COMPETITION.WARNING.multiple': 'These competitions do not match: **{matches}**.',
    'IS.MATCHES_COMPETITION.WARNING.multiple.beginning': 'These competitions do not match: ',
    'IS.MATCHES_COMPETITION.WARNING.multiple.end': '.',
    'IS.MATCHES_COMPETITION.WARNING.single': 'Competition **{name}** does not match.',
    'IS.MATCHES_COMPETITION.WARNING.single.beginning': 'Competition ',
    'IS.MATCHES_COMPETITION.WARNING.single.end': ' does not match.',
    'IS.MATCHES_PYRAMID.PASS.multiple':
        ('These pyramid themes match with the following multipliers: **{matches}**. The final pyramid theme '
         'multiplier is **{finalMultiplier}**. Effective pyramid count for Genius is **{effective}**.'),
    'IS.MATCHES_PYRAMID.PASS.single':
        ('Pyramid theme **{name}** matches with multiplier of **{multiplier}**. Effective pyramid count for '
         'Genius is **{effective}**.'),
    'IS.MATCHES_THEMES.ERROR': 'Theme check error.',
    'IS.MATCHES_THEMES.PASS.multiple':
        ('These themes match with the following multipliers: **{matches}**. The final theme multiplier is '
         '**{finalMultiplier}**.'),
    'IS.MATCHES_THEMES.PASS.single': 'Theme **{name}** matches with multiplier of **{multiplier}**.',
    'IS.MATCHES_THEMES.PENDING': 'Theme check pending.',
    'IS.MATCHES_THEMES.PENDING.multiple':
        ('These themes with the following multipliers will be processed when checking submission: '
         '**{matches}**.'),
    'IS.MATCHES_THEMES.PENDING.single':
        'Theme **{name}** with multiplier of **{multiplier}** will be processed when checking submission.',
    'IS.MATCHES_THEMES.WARNING.multiple': 'These themes do not match with the following multipliers: **{matches}**.',
    'IS.MATCHES_THEMES.WARNING.single': 'Theme **{name}** does not match with multiplier of **{multiplier}**.',
    'IS.MIN_SUBMITTABLE_CHILDREN.FAIL':
        ('Only **{value}** children passed submission tests. At least **{limit}** children must pass '
         'submission tests.'),
    'IS.MIN_SUBMITTABLE_CHILDREN.PASS': '**{value}** children passed submission tests.',
    'IS.NEUTRALIZATION.FAIL': 'Universe **{universe}** does not allow neutralization **{neutralization}**.',
    'IS.NON_PRODUCTIONIZABLE.WARNING': 'This alpha is non-productionizable.',
    'IS.NON_SELF_SUPER_ALPHA.FAIL': 'Non-self SuperAlpha submissions **{value}** reached quota of **{limit}**.',
    'IS.NON_SELF_SUPER_ALPHA.PASS': 'Non-self SuperAlpha submissions **{value}** below quota of **{limit}**.',
    'IS.NON_SELF_SUPER_ALPHA.PENDING': 'Non-self SuperAlpha check pending.',
    'IS.NON_SELF_SUPER_ALPHA.WARNING': 'Non-self SuperAlpha submissions **{value}** reached quota of **{limit}**.',
    'IS.OLD_SIMULATION.FAIL':
        "The simulation doesn't match the current IS period. Please clone this alpha and re-simulate.",
    'IS.OPERATOR_AUTHORIZATION.FAIL': 'Invalid operators: {operators}',
    'IS.OSMOSIS_ALLOCATION.WARNING': 'Daily Osmosis Rank not generated.',
    'IS.POWER_POOL_CORRELATION.ERROR': 'Power Pool correlation check error.',
    'IS.POWER_POOL_CORRELATION.PASS':
        ('Power Pool correlation **{value}** is below cutoff of **{limit}**, or Sharpe is better by 10.0% or '
         'more.'),
    'IS.POWER_POOL_CORRELATION.PENDING': 'Power Pool correlation check pending.',
    'IS.POWER_POOL_CORRELATION.WARNING':
        ('Power Pool correlation **{value}** is above cutoff of **{limit}** and Sharpe not better by 10.0% or '
         'more.'),
    'IS.POWER_POOL_DESCRIPTION_FORMAT.WARNING':
        'Description is not in the correct format for the Power Pool Competition 2025.',
    'IS.POWER_POOL_DESCRIPTION_LENGTH.WARNING': 'Description length **{value}** is below cutoff of **{limit}**.',
    'IS.POWER_POOL_MONTHLY_SUBMISSION.FAIL':
        'Power Pool {scope} monthly submissions **{value}** reached quota of **{limit}**.',
    'IS.POWER_POOL_MONTHLY_SUBMISSION.PASS':
        'Power Pool {scope} monthly submissions **{value}** below quota of **{limit}**.',
    'IS.POWER_POOL_MONTHLY_SUBMISSION.WARNING':
        'Power Pool {scope} monthly submissions **{value}** approaching quota of **{limit}**.',
    'IS.POWER_POOL_MONTHLY_SUBMISSION_REACHED.WARNING':
        'Power Pool {scope} monthly submissions **{value}** reached quota of **{limit}**.',
    'IS.POWER_POOL_SUBMISSION.FAIL': 'Power Pool submissions **{value}** reached quota of **{limit}**.',
    'IS.POWER_POOL_SUBMISSION.PASS': 'Power Pool submissions **{value}** below quota of **{limit}**.',
    'IS.POWER_POOL_SUBMISSION.WARNING': 'Power Pool submissions **{value}** approaching quota of **{limit}**.',
    'IS.POWER_POOL_SUBMISSION_REACHED.WARNING': 'Power Pool submissions **{value}** reached quota of **{limit}**.',
    'IS.PROD_CORRELATION.ERROR': 'Prod correlation check error.',
    'IS.PROD_CORRELATION.FAIL':
        'Prod correlation **{value}** is above cutoff of **{limit}** and Sharpe not better by 10.0% or more.',
    'IS.PROD_CORRELATION.PASS':
        'Prod correlation **{value}** is below cutoff of **{limit}**, or Sharpe is better by 10.0% or more.',
    'IS.PROD_CORRELATION.PENDING': 'Production correlation check pending.',
    'IS.PROD_CORRELATION.WARNING':
        'Prod correlation **{value}** is above cutoff of **{limit}** and Sharpe not better by 10.0% or more.',
    'IS.PURE_POWER_POOL_THEME.FAIL': 'Pure Power Pool submission does not match any Power Pool Theme.',
    'IS.REGION_AUTHORIZATION.FAIL': 'Invalid region: {region}',
    'IS.REGULAR_SUBMISSION.FAIL': 'Alpha submissions **{value}** reached quota of **{limit}**.',
    'IS.REGULAR_SUBMISSION.PASS': 'Alpha submissions **{value}** below quota of **{limit}**.',
    'IS.REGULAR_SUBMISSION.PENDING': 'Alpha submissions quota check pending.',
    'IS.REGULAR_SUBMISSION.WARNING': 'Alpha submissions **{value}** approaching quota of **{limit}**.',
    'IS.REGULAR_SUBMISSION_COMPENSATION.WARNING':
        ('Alpha submissions **{value}** reached compensation quota of **{limit}**; Submission quota is '
         'unlimited.'),
    'IS.REGULAR_SUBMISSION_REACHED.WARNING': 'Alpha submissions **{value}** reached quota of **{limit}**.',
    'IS.REQUIRED_MAX_TRADE.FAIL': 'Max trade is OFF.',
    'IS.REQUIRED_MAX_TRADE.PASS': 'Max trade is ON.',
    'IS.RESEARCHER_RA_SUBMISSION.FAIL': 'Region-agnostic Alpha submissions **{value}** reached quota of **{limit}**.',
    'IS.RESEARCHER_RA_SUBMISSION.PASS': 'Region-agnostic Alpha submissions **{value}** below quota of **{limit}**.',
    'IS.RESEARCHER_RA_SUBMISSION.PENDING': 'Region-agnostic Alpha submissions quota check pending.',
    'IS.RESEARCHER_RA_SUBMISSION.WARNING':
        'Region-agnostic Alpha submissions **{value}** approaching quota of **{limit}**.',
    'IS.RESEARCHER_REGULAR_SUBMISSION.FAIL': 'Researcher Alpha submissions **{value}** reached quota of **{limit}**.',
    'IS.RESEARCHER_REGULAR_SUBMISSION.PASS': 'Researcher Alpha submissions **{value}** below quota of **{limit}**.',
    'IS.RESEARCHER_REGULAR_SUBMISSION.PENDING': 'Researcher Alpha submissions quota check pending.',
    'IS.RESEARCHER_REGULAR_SUBMISSION.WARNING':
        'Researcher Alpha submissions **{value}** approaching quota of **{limit}**.',
    'IS.REVERSION_COMPONENT.WARNING':
        ('Alpha expression includes a reversion component, and we may not accept these alphas in the future. '
         'Try working on different ideas for the alpha.'),
    'IS.RISK_PRECISE.FAIL': 'Risk neutralization is not "Precise".',
    'IS.RISK_PRECISE.PASS': 'Risk neutralization is "Precise".',
    'IS.SELECTION.WARNING': '{message}',
    'IS.SELECTION_DESCRIPTION_LENGTH.FAIL': 'Selection description length **{value}** is below cutoff of **{limit}**.',
    'IS.SELF_CORRELATION.ERROR': 'Self-correlation check error.',
    'IS.SELF_CORRELATION.FAIL':
        'Self-correlation **{value}** is above cutoff of **{limit}** and Sharpe not better by 10.0% or more.',
    'IS.SELF_CORRELATION.PASS':
        ('Self-correlation is **{value}**, which is above cutoff of **{limit}**, or Sharpe is better by 10.0% '
         'or more.'),
    'IS.SELF_CORRELATION.PENDING': 'Self-correlation check pending.',
    'IS.SELF_CORRELATION.WARNING':
        'Self-correlation **{value}** is above cutoff of **{limit}** and Sharpe not better by 10.0% or more.',
    'IS.SELF_SUPER_ALPHA.PASS': 'This alpha is self SuperAlpha.',
    'IS.SUBMISSION_DENIED.FAIL': 'You are not permitted to submit alphas.',
    'IS.SUB_UNIVERSE_SHARPE.ERROR': 'Sub-universe Sharpe check error.',
    'IS.SUB_UNIVERSE_SHARPE.FAIL': 'Sub-universe Sharpe of **{value}** is below cutoff of **{limit}**.',
    'IS.SUB_UNIVERSE_SHARPE.PASS': 'Sub-universe Sharpe of **{value}** is above cutoff of **{limit}**.',
    'IS.SUB_UNIVERSE_SHARPE.WARNING': 'There is a high likelihood of failing the sub-universe check after submission.',
    'IS.SUPER_SUBMISSION.FAIL': 'SuperAlpha submissions **{value}** reached quota of **{limit}**.',
    'IS.SUPER_SUBMISSION.PASS': 'SuperAlpha submissions **{value}** below quota of **{limit}**.',
    'IS.SUPER_SUBMISSION.PENDING': 'SuperAlpha submissions quota check pending.',
    'IS.SUPER_SUBMISSION.WARNING': 'SuperAlpha submissions **{value}** approaching quota of **{limit}**.',
    'IS.SUPER_SUBMISSION_COMPENSATION.WARNING':
        ('SuperAlpha submissions **{value}** reached compensation quota of **{limit}**; Submission quota is '
         'unlimited.'),
    'IS.TECHNICAL_TEST.FAIL': 'Technical test failed.',
    'IS.TECHNICAL_TEST.PASS': 'Technical test passed.',
    'IS.TECHNICAL_TEST.PENDING': 'Technical test is pending.',
    'IS.UNITS.WARNING': '{message}',
    'IS.USA_D1_FUNDAMENTAL.FAIL':
        'USA D1 fundamental data Alphas are not permitted for Inter-IIT Tech Meet participants.',
    'IS.USER_DEFINED_FIELDS.FAIL': "Alphas containing user defined fields and can't be submitted.",
    'OS.BIAS.FAIL': 'Bias check failed.',
    'OS.BIAS.PASS': 'Bias check passed.',
    'OS.BIAS.PENDING': 'Bias check pending.',
    'OS.CHECKPOINT.FAIL': 'Checkpoint check failed.',
    'OS.CHECKPOINT.PASS': 'Checkpoint check passed.',
    'OS.CHECKPOINT.PENDING': 'Checkpoint check pending.',
    'OS.COMPENSATED_ALPHA.WARNING': 'This alpha is not compensated.',
    'OS.CONCENTRATED_WEIGHT.ERROR': 'Weight check error.',
    'OS.CONCENTRATED_WEIGHT.FAIL': 'Weight is too strongly concentrated or too few instruments are assigned weight.',
    'OS.CONCENTRATED_WEIGHT.PASS': 'Weight is well distributed over instruments.',
    'OS.CONCENTRATED_WEIGHT.PENDING': 'Weight concentration test pending.',
    'OS.CONCENTRATED_WEIGHT.WITH_VALUES.FAIL':
        'Weight concentration **{value}** is above cutoff of **{limit}** on **{date}**.',
    'OS.CONCENTRATED_WEIGHT.WITH_VALUES.PASS':
        'Weight is well distributed over instruments. **{value}** is below cutoff of **{limit}**',
    'OS.DRAWDOWN.FAIL': 'Drawdown check failed.',
    'OS.DRAWDOWN.PASS': 'Drawdown check passed.',
    'OS.DRAWDOWN.PENDING': 'Drawdown check pending.',
    'OS.DUPLICATE_ALPHA.FAIL': 'Duplicate Alpha.',
    'OS.DUPLICATE_ALPHA.PASS': 'Duplicate Alpha.',
    'OS.DUPLICATE_ALPHA.PENDING': 'Duplicate Alpha.',
    'OS.IMBALANCE.ERROR': 'Imbalance check error.',
    'OS.IMBALANCE.FAIL':
        'Imbalance of **{value}** is above cutoff of **{limit}** on **{date}**. Check neutralization setting.',
    'OS.IMBALANCE.PASS': 'Imbalance of **{value}** is below cutoff of **{limit}**.',
    'OS.IMBALANCE.PENDING': 'Imbalance check pending.',
    'OS.INVALID_CODE.FAIL': 'Invalid Alpha code.',
    'OS.INVALID_CODE.PASS': 'Invalid Alpha code.',
    'OS.INVALID_CODE.PENDING': 'Invalid Alpha code.',
    'OS.INVALID_UNIVERSE.FAIL': 'Invalid Universe.',
    'OS.INVALID_UNIVERSE.PASS': 'Invalid Universe.',
    'OS.INVALID_UNIVERSE.PENDING': 'Invalid Universe.',
    'OS.IS_SHARPE.FAIL': 'IS Sharpe check failed.',
    'OS.IS_SHARPE.PASS': 'IS sharpe check passed.',
    'OS.IS_SHARPE.PENDING': 'IS sharpe check pending.',
    'OS.MEMORY_USAGE.FAIL': 'Memory usage check failed.',
    'OS.MEMORY_USAGE.PASS': 'Memory usage check passed.',
    'OS.MEMORY_USAGE.PENDING': 'Memory usage check pending.',
    'OS.NEW_HIGH.FAIL': 'New high check failed.',
    'OS.NEW_HIGH.PASS': 'New high check passed.',
    'OS.NEW_HIGH.PENDING': 'New high check pending.',
    'OS.OTHERS.FAIL': 'Other check failed.',
    'OS.OTHERS.PASS': 'Other check passed.',
    'OS.OTHERS.PENDING': 'Other check pending',
    'OS.PRE_CLOSE_SHARPE.FAIL': 'Pre-close Sharpe check failed.',
    'OS.PRE_CLOSE_SHARPE.PASS': 'Pre-close sharpe check passed.',
    'OS.PRE_CLOSE_SHARPE.PENDING': 'Pre-close sharpe check pending.',
    'OS.PROD_CORRELATION.FAIL': 'Prod correlation check failed.',
    'OS.PROD_CORRELATION.PASS': 'Prod correlation check passed.',
    'OS.PROD_CORRELATION.PENDING': 'Production correlation check pending.',
    'OS.RANK_SHARPE.FAIL': 'Rank Sharpe check failed.',
    'OS.RANK_SHARPE.PASS': 'Rank sharpe check passed.',
    'OS.RANK_SHARPE.PENDING': 'Rank sharpe check pending.',
    'OS.SELF_CORRELATION.FAIL': 'Self-correlation check failed.',
    'OS.SELF_CORRELATION.PASS': 'Self-correlation check passed.',
    'OS.SELF_CORRELATION.PENDING': 'Self-correlation check pending.',
    'OS.SHARPE.FAIL': 'Sharpe check failed.',
    'OS.SHARPE.PASS': 'Sharpe check passed.',
    'OS.SHARPE.PENDING': 'Sharpe check pending.',
    'OS.SUB_UNIVERSE_SHARPE.FAIL': 'Sub-universe Sharpe check failed.',
    'OS.SUB_UNIVERSE_SHARPE.PASS': 'Sub-universe Sharpe check passed.',
    'OS.SUB_UNIVERSE_SHARPE.PENDING': 'Sub-universe Sharpe check pending.',
    'OS.SUPER_UNIVERSE_SHARPE.FAIL': 'Super-universe Sharpe check failed.',
    'OS.SUPER_UNIVERSE_SHARPE.PASS': 'Super-universe Sharpe check passed.',
    'OS.SUPER_UNIVERSE_SHARPE.PENDING': 'Super-universe Sharpe check pending.',
    'OS.TECHNICAL_TEST.FAIL': 'Technical test failed.',
    'OS.TECHNICAL_TEST.PASS': 'Technical test passed.',
    'OS.TECHNICAL_TEST.PENDING': 'Technical test is pending.',
    'OS.WEIGHT.FAIL': 'Weight check failed.',
    'OS.WEIGHT.PASS': 'Weight check passed.',
    'OS.WEIGHT.PENDING': 'Weight check pending.',
}


# ── figures ──────────────────────────────────────────────────────────────────────────────────


def is_num(value: Any) -> bool:
    """A finite number, as JavaScript's ``typeof value === 'number' && isFinite(value)`` (a bool is not one)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _grouped(value: float, digits: int, *, min_digits: int | None = None) -> str:
    """``Intl.NumberFormat('en-US')``: thousands separators, half away from zero, at most ``digits`` decimals
    (at least ``min_digits``, default all of them)."""
    exact = Decimal(value) if isinstance(value, int) else Decimal(repr(float(value)))
    rounded = exact.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    text = f"{rounded:,.{digits}f}"
    keep = digits if min_digits is None else min_digits
    if digits > keep and "." in text:
        whole, frac = text.split(".")
        frac = frac.rstrip("0").ljust(keep, "0")
        text = f"{whole}.{frac}" if frac else whole
    if text.startswith("-") and not any(ch in "123456789" for ch in text):
        text = text[1:]                     # Intl prints 0, not -0, for a tiny negative
    return text


def brain_value(value: float, kind: str, in_check: bool = False) -> str:
    """A figure as BRAIN prints it: a percent x100 with ``%``, a permyriad x10 000 with `` bps``, else two decimals;
    inside a check message a whole number drops its decimals."""
    scaled = value * 100 if kind == "percent" else value * 10_000 if kind == "permyriad" else value
    text = _grouped(scaled, 0) if in_check and scaled % 1 == 0 else _grouped(scaled, 2)
    return f"{text}%" if kind == "percent" else f"{text} bps" if kind == "permyriad" else text


def cell(value: Any, kind: str) -> str:
    """A table cell: blank where BRAIN has no figure, counts and labels as sent."""
    if not is_num(value):
        return "" if value is None else js_string(value)
    return _js_number(value) if kind == "integer" else brain_value(value, kind)


def _js_number(value: float) -> str:
    """``Number.prototype.toString``: 2 not 2.0, 1.5 as 1.5."""
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return repr(float(value))


def js_string(value: Any) -> str:
    """``String(value)`` as JavaScript writes it, so a filled template reads as BRAIN's own page does."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if is_num(value):
        return _js_number(value)
    if isinstance(value, (list, tuple)):
        return ",".join("" if v is None else js_string(v) for v in value)
    if isinstance(value, dict):
        return "[object Object]"
    return str(value)


def _field(record: dict[str, Any], key: str) -> str:
    """``${record[key]}`` in a JavaScript template: a missing key reads ``undefined``."""
    return js_string(record[key]) if key in record else "undefined"


def _truthy(value: Any) -> bool:
    """JavaScript truthiness: an empty list or object is true; 0, '', NaN and null are false."""
    if isinstance(value, (list, tuple, dict)):
        return True
    if isinstance(value, float) and math.isnan(value):
        return False
    return bool(value)


def _moment(text: str) -> datetime | None:
    """An ISO date or time as a moment; a bare day is UTC midnight, as JavaScript parses it."""
    try:
        parsed = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _shown(value: Any) -> str:
    """How a value prints inside a message: a day as M/D/YYYY in UTC, a number with up to three decimals."""
    if isinstance(value, datetime):
        day = value.astimezone(timezone.utc)
        return f"{day.month}/{day.day}/{day.year}"
    if is_num(value):
        return _grouped(value, 3, min_digits=0)
    if value is None:
        return ""
    return js_string(value)


# ── messages ─────────────────────────────────────────────────────────────────────────────────

#: Each check's figures as the metric they measure.
MEASURES: dict[str, str] = {
    "LOW_FITNESS": "decimal",
    "LOW_SHARPE": "decimal",
    "IS_LADDER_SHARPE": "decimal",
    "LOW_ROBUST_UNIVERSE_SHARPE": "decimal",
    "HIGH_TURNOVER": "percent",
    "HT_TURNOVER": "percent",
    "HT_INVESTABLE_MAX_TRADE_TURNOVER": "percent",
    "HT_INVESTABLE_MAX_POSITION_TURNOVER": "percent",
    "LOW_TURNOVER": "percent",
    "LOW_ROBUST_UNIVERSE_RETURNS": "percent",
    "LOW_RETURNS": "percent",
    "IMBALANCE": "percent",
    "CONCENTRATED_WEIGHT": "percent",
}

MATCHES = frozenset({"MATCHES_THEMES", "MATCHES_COMPETITION", "MATCHES_PYRAMID", "MATCHES_CLASSIFICATION"})

#: Result groups in the order BRAIN lists them.
RESULT_ORDER = ("PASS", "FAIL", "WARNING", "ERROR", "PENDING")

_BOLD = re.compile(r"(\*\*.*?\*\*)")
_FIELD = re.compile(r"\{(\w+)\}")


@dataclass(frozen=True)
class Part:
    """A run of a message: plain, bold (a figure) or a link (a competition)."""

    text: str
    bold: bool = False
    href: str | None = None


def message(key: str, values: dict[str, Any] | None = None) -> list[Part]:
    """One of BRAIN's templates filled in, its ``**bold**`` runs kept apart; [] when BRAIN has no such template."""
    values = values or {}
    template = CHECK_MESSAGES.get(f"{key}.null") if "value" in values and values["value"] is None else None
    if template is None:
        template = CHECK_MESSAGES.get(key)
    if template is None:
        return []
    parts: list[Part] = []
    for run in _BOLD.split(template):
        if not run:
            continue
        bold = len(run) > 4 and run.startswith("**") and run.endswith("**")
        raw = run[2:-2] if bold else run
        parts.append(Part(_FIELD.sub(lambda m: _shown(values.get(m.group(1))), raw), bold=bold))
    return parts


def _records(raw: Any) -> list[dict[str, Any]]:
    return [x for x in raw if isinstance(x, dict)] if isinstance(raw, list) else []


def _competition(match: dict[str, Any]) -> Part:
    name = match.get("name")
    return Part("" if name is None else js_string(name), href=f"{PLATFORM_COMPETITIONS_URL}{_field(match, 'id')}")


def check_parts(check: dict[str, Any], stage: str = "IS") -> list[Part]:
    """BRAIN's sentence for one check, in the stage's own wording (``IS`` or ``OS``), as runs."""
    stage = "OS" if str(stage).upper() == "OS" else "IS"
    name = str(check.get("name"))
    result = str(check.get("result") or "PENDING")
    key = f"{stage}.{name}"
    c: dict[str, Any] = dict(check)
    if is_num(c.get("ratio")):
        c["ratio"] = brain_value(c["ratio"], "percent", True)
    measure = MEASURES.get(name)
    for k in ("value", "limit", "max", "min"):
        if k not in c:
            continue
        v = c[k]
        c[k] = brain_value(v, measure, True) if measure and is_num(v) else js_string(v) if name in MATCHES else v
    for k in ("startDate", "endDate", "date"):
        if isinstance(c.get(k), str):
            c[k] = _moment(c[k]) or c[k]

    def said(parts: list[Part]) -> list[Part]:
        return parts or [Part(f"{name} {result}")]

    if name == "LOW_ROBUST_UNIVERSE_SHARPE" and c.get("ratio") is not None:
        return said(message(f"{key}.WITH_RATIO.{result}", c))
    if "value" not in c:
        c["value"] = "none"

    listed = name in MATCHES and (
        result in ("PASS", "WARNING")
        or (name == "MATCHES_THEMES" and result == "PENDING" and _truthy(c.get("themes")))
        or (name == "MATCHES_COMPETITION" and result == "PENDING" and _truthy(c.get("competitions"))))
    if listed:
        if name == "MATCHES_THEMES" and _truthy(c.get("themes")):
            matches = _records(c.get("themes"))
        elif name == "MATCHES_COMPETITION" and _truthy(c.get("competitions")):
            matches = _records(c.get("competitions"))
        elif name == "MATCHES_PYRAMID" and _truthy(c.get("pyramids")):
            matches = _records(c.get("pyramids"))
        elif name == "MATCHES_CLASSIFICATION" and isinstance(c.get("value"), list):
            matches = [{"name": v} for v in c["value"]]
        else:
            matches = _records(c.get("matched") if result == "PASS" else c.get("unmatched"))
        at = f"{key}.{result}"
        first = matches[0] if matches else None
        if first is not None and len(matches) == 1:
            if name == "MATCHES_COMPETITION" and _truthy(first.get("id")):
                return [*message(f"{at}.single.beginning"), _competition(first), *message(f"{at}.single.end")]
            multiplier = _js_number(first["multiplier"]) if is_num(first.get("multiplier")) else None
            return said(message(f"{at}.single", {**c, **first, "multiplier": multiplier}))
        if first is not None and len(matches) > 1:
            if name == "MATCHES_COMPETITION" and _truthy(first.get("id")):
                links: list[Part] = []
                for i, m in enumerate(matches):
                    if i:
                        links.append(Part(", "))
                    links.append(_competition(m))
                return [*message(f"{at}.multiple.beginning"), *links, *message(f"{at}.multiple.end")]
            if name == "MATCHES_CLASSIFICATION":
                return said(message(f"{at}.multiple", {"value": "; ".join(_field(m, "name") for m in matches)}))
            if is_num(first.get("multiplier")):
                filled = {"matches": "; ".join(f"{_field(m, 'name')} — {_field(m, 'multiplier')}" for m in matches),
                          "finalMultiplier": c.get("multiplier"), "effective": c.get("effective")}
            else:
                filled = {"matches": ", ".join(_field(m, "name") for m in matches)}
            return said(message(f"{at}.multiple", filled))

    if name == "DATA_DIVERSITY" and result in ("WARNING", "FAIL"):
        fields = c.get("fields") if isinstance(c.get("fields"), list) else []
        return said(message(f"{stage}.DATA_DIVERSITY.NOT_PASSED", {
            "dataCategoryName": c.get("dataCategoryName"),
            "field": ", ".join(f'"{js_string(f)}"' for f in fields)}))
    if name == "DATA_SET_AUTHORIZATION" and result == "FAIL":
        return said(message(f"{key}.{result}", {"datasets": _flat_join(c.get("datasets"))}))
    if name == "OPERATOR_AUTHORIZATION" and result == "FAIL":
        return said(message(f"{key}.{result}", {"operators": _flat_join(c.get("operators"))}))
    value = c.get("value")
    weighed = name == "CONCENTRATED_WEIGHT" and ((_truthy(value) and value != "none") or _truthy(c.get("limit")))
    return said(message(f"{key}{'.WITH_VALUES' if weighed else ''}.{result}", c))


def _flat_join(value: Any) -> str:
    """``[value].flat().join(', ')``: a list joined, a lone value as itself, null as nothing."""
    items = value if isinstance(value, list) else [value]
    return ", ".join("" if v is None else js_string(v) for v in items)


def check_message(check: dict[str, Any], stage: str = "IS", *, markdown: bool = True) -> str:
    """BRAIN's sentence for one check. With ``markdown`` its figures stay bold as ``**x**`` and competitions read
    ``[name](url)``; without, plain text."""
    out = []
    for part in check_parts(check, stage):
        if not markdown:
            out.append(part.text)
        elif part.href:
            out.append(f"[{part.text}]({part.href})")
        elif part.bold:
            out.append(f"**{part.text}**")
        else:
            out.append(part.text)
    return "".join(out)


def parts_html(parts: list[Part]) -> str:
    """Runs as safe HTML (BRAIN's text is escaped): bold figures in ``<strong>``, competitions as links to BRAIN."""
    out = []
    for part in parts:
        text = html.escape(part.text)
        if part.href:
            out.append(f'<a href="{html.escape(part.href)}" target="_blank" rel="noopener" class="ag-link">{text}</a>')
        elif part.bold:
            out.append(f'<strong class="ag-mono">{text}</strong>')
        else:
            out.append(text)
    return "".join(out)


def check_html(check: dict[str, Any], stage: str = "IS") -> str:
    """BRAIN's sentence for one check as safe HTML."""
    return parts_html(check_parts(check, stage))


def by_result(checks: list[dict[str, Any]] | None) -> list[tuple[str, list[dict[str, Any]]]]:
    """Checks grouped by result, groups in BRAIN's order (unknown results last), each in the order sent."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for check in checks or []:
        if isinstance(check, dict):
            groups.setdefault(str(check.get("result") or "PENDING"), []).append(check)

    def rank(result: str) -> int:
        return RESULT_ORDER.index(result) if result in RESULT_ORDER else len(RESULT_ORDER)

    return sorted(groups.items(), key=lambda item: rank(item[0]))
