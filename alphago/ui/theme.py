"""Dark theme modelled on the reference dashboard: deep navy surfaces, teal metrics, violet accents."""

from __future__ import annotations

from nicegui import ui

CSS = r"""
:root {
  --ag-bg: #070b14;
  --ag-surface: #0d1422;
  --ag-surface-2: #111a2b;
  --ag-border: #1e2b40;
  --ag-border-strong: #2a3b57;
  --ag-text: #e6edf7;
  --ag-muted: #8b98ad;
  --ag-teal: #5eead4;
  --ag-cyan: #7dd3fc;
  --ag-violet: #a78bfa;
  --ag-pink: #f472b6;
  --ag-amber: #fbbf24;
  --ag-green: #4ade80;
  --ag-red: #f87171;
  --ag-mono: "JetBrains Mono", "Cascadia Mono", Consolas, monospace;
}
body, .q-page, .nicegui-content { background: var(--ag-bg) !important; color: var(--ag-text); }
body { font-family: Inter, "Segoe UI", system-ui, sans-serif; }
.nicegui-content { padding: 0 !important; }
.ag-mono { font-family: var(--ag-mono); }
.ag-muted { color: var(--ag-muted); }
.ag-teal { color: var(--ag-teal); }
.ag-cyan { color: var(--ag-cyan); }
.ag-violet { color: var(--ag-violet); }
.ag-green { color: var(--ag-green); }
.ag-red { color: var(--ag-red); }
.ag-amber { color: var(--ag-amber); }
.ag-pink { color: var(--ag-pink); }
.ag-seg { border: 1px solid var(--ag-border-strong); border-radius: 8px; background: var(--ag-surface-2); }
.ag-seg .q-btn { color: #c9d4e5; padding: 2px 12px; }
.ag-stop-all { border-radius: 8px; }
.q-btn .q-btn__content { flex-wrap: nowrap; white-space: nowrap; }
.ag-drawer .q-drawer__content { display: flex; flex-direction: column; }
.ag-nav-badge { margin-left: auto; font-family: var(--ag-mono); font-size: 12px; color: var(--ag-teal); }
.ag-warn-box { border: 1px solid #5a4a1a; background: rgba(80,60,10,.15); border-radius: 10px; padding: 10px 14px;
  color: var(--ag-amber); }
.ag-bad-box { border: 1px solid #5a1f2a; background: rgba(80,20,30,.15); border-radius: 10px; padding: 10px 14px; }

/* header + drawer */
.ag-header { background: rgba(7,11,20,0.92) !important; border-bottom: 1px solid var(--ag-border); backdrop-filter: blur(6px); }
.ag-drawer { background: var(--ag-bg) !important; border-right: 1px solid var(--ag-border); }
.ag-nav-item { border-radius: 10px; padding: 7px 12px; color: #c9d4e5; cursor: pointer; transition: background .15s;
  border-left: 3px solid transparent; }
.ag-nav-item:hover { background: var(--ag-surface-2); }
.ag-nav-item.active { background: linear-gradient(90deg, rgba(167,139,250,0.20), rgba(167,139,250,0.04));
  border-left: 3px solid var(--ag-violet); color: #fff; }
.ag-chip { border: 1px solid #1f4a4a; background: rgba(20, 60, 60, 0.25); color: var(--ag-teal);
  border-radius: 8px; padding: 5px 12px; font-size: 13px; white-space: nowrap; }
.ag-chip.warn { border-color: #5a4a1a; color: var(--ag-amber); background: rgba(80,60,10,.2); }
.ag-chip.bad { border-color: #5a1f2a; color: var(--ag-red); background: rgba(80,20,30,.2); }

/* cards */
.ag-card { background: linear-gradient(180deg, var(--ag-surface) 0%, #0b1220 100%) !important;
  border: 1px solid var(--ag-border); border-radius: 14px !important; box-shadow: none !important; color: var(--ag-text); }
.ag-card-violet { border-color: #3b2f6b; }
.ag-tile { background: var(--ag-surface-2) !important; border: 1px solid var(--ag-border-strong); border-radius: 10px !important;
  box-shadow: none !important; padding: 14px 16px; }
.ag-kpi { font-family: var(--ag-mono); font-size: 34px; font-weight: 500; letter-spacing: 1px; }
.ag-kpi-sm { font-family: var(--ag-mono); font-size: 22px; }
.ag-title { font-size: 20px; font-weight: 600; }
.ag-section-title { font-size: 15px; font-weight: 600; }
.ag-link { color: var(--ag-violet); text-decoration: underline; cursor: pointer; font-size: 13px; }
.ag-badge { border-radius: 6px; padding: 1px 8px; font-size: 12px; border: 1px solid var(--ag-border-strong); }
.ag-badge.pass { color: var(--ag-green); border-color: #1d4d33; background: rgba(30,90,50,.2); }
.ag-badge.fail { color: var(--ag-red); border-color: #5a1f2a; background: rgba(90,20,30,.2); }
.ag-badge.pending, .ag-badge.warning { color: var(--ag-amber); border-color: #5a4a1a; background: rgba(90,70,10,.2); }
.ag-badge.stopped { color: #fda4af; background: rgba(120,20,40,.35); border-color: #6b2333; }
.ag-badge.running { color: var(--ag-green); background: rgba(20,90,50,.3); border-color: #1d4d33; }
.ag-feed-item { border-bottom: 1px solid var(--ag-border); padding: 8px 2px; }
.ag-src { border: 1px solid var(--ag-border-strong); border-radius: 6px; padding: 0 8px; font-size: 12px; color: #c9d4e5; }

/* quasar inputs on dark */
.q-field__native, .q-field__input, .q-select__dropdown-icon { color: var(--ag-text) !important; }
.q-field--outlined .q-field__control:before { border-color: var(--ag-border-strong) !important; }
.q-tab-panels, .q-tab-panel { background: transparent !important; }
.q-table__card, .q-table { background: var(--ag-surface) !important; color: var(--ag-text) !important; }
.q-menu, .q-dialog__inner > div { background: var(--ag-surface) !important; color: var(--ag-text); }
.q-expansion-item { border: 1px solid var(--ag-border); border-radius: 10px; }

/* ag-grid dark */
.nicegui-aggrid { --ag-background-color: #0d1422; --ag-header-background-color: #111a2b;
  --ag-odd-row-background-color: #0f1828; --ag-border-color: #1e2b40; --ag-foreground-color: #e6edf7;
  --ag-row-hover-color: #16233a; --ag-selected-row-background-color: #231c3f;
  --ag-font-family: Inter, "Segoe UI", sans-serif; --ag-font-size: 13px; }
.ag-num { font-family: var(--ag-mono); }
"""

SIDEBAR_WIDTH = 270


def apply_theme() -> None:
    ui.dark_mode(True)
    ui.colors(primary="#8b5cf6", secondary="#5eead4", accent="#f472b6", positive="#4ade80",
              negative="#f87171", warning="#fbbf24", info="#7dd3fc", dark="#0d1422")
    ui.add_css(CSS)
