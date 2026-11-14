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
.ag-header { background: rgba(7,11,20,0.92) !important; border-bottom: 1px solid var(--ag-border);
             backdrop-filter: blur(6px); }
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
.ag-tile { background: var(--ag-surface-2) !important; border: 1px solid var(--ag-border-strong);
           border-radius: 10px !important;
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

/* core slots (DESIGN.md: one tile per simulation core; idle is a hollow dashed socket, never a grey button) */
:root { --ag-status-idle: #3a4a66; --ag-status-queued: #2a3b57; --ag-status-running: #7c6cf0;
        --ag-status-warning: #d9a21b; }
.ag-cores { gap: 6px; border: 1px solid var(--ag-border); background: var(--ag-surface); border-radius: 8px;
            padding: 4px; overflow-x: auto; scrollbar-width: none; }
.ag-core { position: relative; width: 28px; height: 28px; border-radius: 4px; display: flex; align-items: center;
           justify-content: center; overflow: hidden; font-family: var(--ag-mono); font-size: 11px;
           font-weight: 500; user-select: none; transition: background .2s; }
.ag-core.idle { border: 1px dashed var(--ag-status-idle); background: var(--ag-surface); color: var(--ag-muted); }
.ag-core.pending { background: var(--ag-status-queued); color: var(--ag-text); }
.ag-core.running { background: var(--ag-status-running); color: #fff; }
.ag-core.warning { background: var(--ag-status-warning); color: #111; }
.ag-core.first { border-top-right-radius: 0; border-bottom-right-radius: 0; margin-right: -5px; }
.ag-core.mid { border-radius: 0; margin-right: -5px; }
.ag-core.last { border-top-left-radius: 0; border-bottom-left-radius: 0; }
.ag-core-fill { position: absolute; left: 0; bottom: 0; height: 2px; width: 0; background: var(--ag-text);
                transition: width .4s; }
.ag-core-label { font-family: var(--ag-mono); font-size: 12px; font-weight: 500; padding: 4px 8px; border-radius: 4px;
                 border: 1px solid var(--ag-border-strong); background: var(--ag-surface-2); min-width: 44px;
                 text-align: center; }
.ag-core-label.running { border-color: var(--ag-status-running); }
.ag-core-label.warning { border-color: var(--ag-status-warning); }
.ag-core-label.idle { border-style: dashed; color: var(--ag-muted); }
.ag-batch-key { border: 1px solid var(--ag-border); border-radius: 8px; padding: 3px 12px; min-height: 28px;
                background: var(--ag-surface); }
.ag-batch-key.on { border-color: var(--ag-border-strong); background: var(--ag-surface-2); }
.ag-progress { width: 64px; height: 6px; border-radius: 99px; background: var(--ag-border); overflow: hidden; }
.ag-progress-fill { height: 100%; width: 0; background: var(--ag-status-running); transition: width .4s; }
.ag-cell { height: 30px; border-radius: 4px; display: flex; align-items: center; justify-content: center; }
.ag-cell.empty { border: 1px solid var(--ag-border); background: var(--ag-bg); }
.ag-cell.pending { border: 1px solid var(--ag-border); background: var(--ag-surface-2); }
.ag-cell.running { border: 1px solid var(--ag-border-strong); background: rgba(230,237,247,.06); }
.ag-cell-dot { width: 6px; height: 6px; border-radius: 99px; background: var(--ag-status-running); }

/* sidebar sections */
.ag-nav-section { border: none !important; }
.ag-nav-section > .q-expansion-item__container > .q-item { min-height: 28px; padding: 2px 8px; color: var(--ag-muted);
  font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }
.ag-nav-section .q-expansion-item__content { padding-bottom: 4px; }

/* command palette */
.ag-palette .q-field__native { font-size: 15px; }
.ag-palette-item { border-radius: 6px; padding: 6px 10px; gap: 10px; cursor: pointer; color: #c9d4e5; }
.ag-palette-item:hover { background: var(--ag-surface-2); }
.ag-palette-item.selected { background: rgba(167,139,250,0.18); color: #fff; }
.ag-kbd { font-family: var(--ag-mono); font-size: 11px; color: var(--ag-muted); border: 1px solid var(--ag-border-strong);
          border-radius: 4px; padding: 0 6px; }

/* alpha page */
.ag-panel-title { font-size: 15px; font-weight: 600; }
.ag-figure-label { color: var(--ag-muted); font-size: 12px; }
.ag-figure { font-family: var(--ag-mono); font-size: 20px; white-space: nowrap; }
.ag-dot { width: 6px; height: 6px; border-radius: 99px; flex-shrink: 0; margin-top: 8px; }
.ag-dot.PASS { background: var(--ag-green); } .ag-dot.FAIL, .ag-dot.ERROR { background: var(--ag-red); }
.ag-dot.WARNING { background: var(--ag-amber); } .ag-dot.PENDING { background: var(--ag-muted); }
.ag-table { width: 100%; border-collapse: collapse; font-size: 13px; white-space: nowrap; }
.ag-table th { color: var(--ag-muted); font-weight: 500; padding: 5px 8px; text-align: right;
               background: var(--ag-surface-2); border-bottom: 1px solid var(--ag-border); }
.ag-table td { font-family: var(--ag-mono); padding: 5px 8px; text-align: right; border-bottom: 1px solid #16233a; }
.ag-table th:first-child, .ag-table td:first-child { text-align: left; }
.ag-table td.stage-TRAIN { border-left: 2px solid var(--ag-status-running); }
.ag-table td.stage-TEST { border-left: 2px solid var(--ag-status-warning); }
.ag-table td.stage-OS { border-left: 2px solid var(--ag-green); }
.ag-table-wrap { overflow-x: auto; border: 1px solid var(--ag-border); border-radius: 8px; }
"""

SIDEBAR_WIDTH = 270


def apply_theme() -> None:
    ui.dark_mode(True)
    ui.colors(primary="#8b5cf6", secondary="#5eead4", accent="#f472b6", positive="#4ade80",
              negative="#f87171", warning="#fbbf24", info="#7dd3fc", dark="#0d1422")
    ui.add_css(CSS)
