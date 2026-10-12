"""Generate integration/functions.html call map from AST."""
from __future__ import annotations

import ast
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "integration"
SKIP = {"__init__.py", "config.py"}
AMBIGUOUS = {"_prepare_matrix", "_require_columns", "_round_numbers"}


def collect() -> tuple[list[dict], list[dict]]:
    mods: dict[str, dict] = {}
    edges_raw: list[tuple[str, str]] = []

    for path in sorted(ROOT.rglob("*.py")):
        if path.name in SKIP or "tests" in path.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        mod = rel[:-3].replace("/", ".")
        src = path.read_text(encoding="utf-8")
        lines = src.splitlines()
        tree = ast.parse(src)
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            tag = "unknown"
            for i in range(node.lineno - 2, max(-1, node.lineno - 8), -1):
                if i < 0:
                    break
                t = lines[i].strip()
                if t in ("#new", "#reviewed"):
                    tag = t[1:]
                    break
                if t.startswith(("def ", "class ", "async def")):
                    break
            q = f"{mod}.{node.name}"
            mods[q] = {
                "id": q,
                "module": mod,
                "name": node.name,
                "tag": tag,
                "lineno": node.lineno,
                "file": rel,
            }
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Call):
                    continue
                fn = sub.func
                callee = None
                if isinstance(fn, ast.Name):
                    callee = fn.id
                elif isinstance(fn, ast.Attribute):
                    callee = fn.attr
                if callee:
                    edges_raw.append((q, callee))

    by_name: dict[str, list[str]] = defaultdict(list)
    for q, m in mods.items():
        by_name[m["name"]].append(q)

    resolved: set[tuple[str, str]] = set()
    for caller, callee in edges_raw:
        cands = by_name.get(callee, [])
        if not cands:
            continue
        cmod = mods[caller]["module"]
        same = [c for c in cands if mods[c]["module"] == cmod]
        targets = same if (callee in AMBIGUOUS or same) else cands
        if callee in AMBIGUOUS:
            targets = same
        for t in targets:
            if t != caller:
                resolved.add((caller, t))

    nodes = sorted(mods.values(), key=lambda x: (x["module"], x["name"]))
    edges = [{"from": a, "to": b} for a, b in sorted(resolved)]
    return nodes, edges


def main() -> None:
    nodes, edges = collect()
    data_js = json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False)

    html_doc = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Querulus Integration — карта функций</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet" />
  <style>
    :root {{
      --bg: #0f1419;
      --bg-elev: #1a222c;
      --bg-card: #222b36;
      --line: #3a4654;
      --text: #e8eef4;
      --muted: #9aabbc;
      --accent: #3db88a;
      --accent-dim: #2a7a5c;
      --main: #4a9fd8;
      --second: #e0a04a;
      --new: #5ec8a0;
      --reviewed: #7a8fa3;
      --hot: #e8c36a;
      --caller: #6bb3e8;
      --callee: #e0a04a;
      --mono: "IBM Plex Mono", ui-monospace, monospace;
      --sans: "IBM Plex Sans", system-ui, sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      font-family: var(--sans);
      background:
        radial-gradient(ellipse 80% 50% at 10% -10%, #1a3a32 0%, transparent 55%),
        radial-gradient(ellipse 60% 40% at 100% 0%, #1a2a3a 0%, transparent 50%),
        var(--bg);
      color: var(--text);
      line-height: 1.5;
      font-size: 15px;
    }}
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 2rem 1.25rem 3.5rem; }}
    header {{
      margin-bottom: 1.5rem;
      border-bottom: 1px solid var(--line);
      padding-bottom: 1.25rem;
    }}
    header .eyebrow {{
      font-family: var(--mono);
      font-size: 0.75rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent);
      margin: 0 0 0.5rem;
    }}
    h1 {{
      font-size: clamp(1.45rem, 2.8vw, 1.95rem);
      font-weight: 600;
      margin: 0 0 0.5rem;
      letter-spacing: -0.02em;
    }}
    header p {{ margin: 0; color: var(--muted); max-width: 48rem; }}
    header a {{ color: var(--main); }}
    header code {{
      font-family: var(--mono);
      font-size: 0.85em;
      color: var(--accent);
    }}
    .toolbar {{
      display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: center;
      margin: 1.25rem 0 1rem;
    }}
    .toolbar input[type="search"] {{
      flex: 1 1 220px;
      min-width: 180px;
      background: var(--bg-elev);
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--text);
      padding: 0.55rem 0.75rem;
      font-family: var(--mono);
      font-size: 0.85rem;
    }}
    .toolbar input:focus {{ outline: 1px solid var(--accent-dim); }}
    .chip {{
      font-family: var(--mono);
      font-size: 0.72rem;
      padding: 0.4rem 0.65rem;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: var(--bg-elev);
      color: var(--muted);
      cursor: pointer;
      user-select: none;
    }}
    .chip.on {{ color: var(--text); border-color: var(--accent-dim); background: rgba(61,184,138,0.12); }}
    .chip.tag-new.on {{ color: var(--new); border-color: #2a7a5c; }}
    .chip.tag-reviewed.on {{ color: #b8c8d6; border-color: #4a5a6a; }}
    .stats {{
      font-family: var(--mono);
      font-size: 0.75rem;
      color: var(--muted);
      margin-left: auto;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 1rem;
      align-items: start;
    }}
    @media (max-width: 900px) {{ .layout {{ grid-template-columns: 1fr; }} }}
    .panel {{
      background: var(--bg-elev);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .panel-h {{
      padding: 0.7rem 1rem;
      border-bottom: 1px solid var(--line);
      font-weight: 600;
      font-size: 0.92rem;
      display: flex; justify-content: space-between; align-items: center;
    }}
    .panel-h span {{ font-family: var(--mono); font-size: 0.7rem; color: var(--muted); font-weight: 500; }}
    .mod {{ border-bottom: 1px solid var(--line); }}
    .mod:last-child {{ border-bottom: none; }}
    .mod-h {{
      padding: 0.55rem 1rem;
      font-family: var(--mono);
      font-size: 0.78rem;
      color: var(--main);
      background: rgba(74,159,216,0.06);
      display: flex; justify-content: space-between;
    }}
    .mod-h.second {{ color: var(--second); background: rgba(224,160,74,0.06); }}
    .mod-body {{ padding: 0.45rem 0.65rem 0.7rem; display: flex; flex-wrap: wrap; gap: 0.35rem; }}
    .fn {{
      font-family: var(--mono);
      font-size: 0.72rem;
      padding: 0.28rem 0.5rem;
      border-radius: 4px;
      border: 1px solid transparent;
      background: var(--bg-card);
      color: var(--text);
      cursor: pointer;
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .fn.new {{ box-shadow: inset 3px 0 0 var(--new); }}
    .fn.reviewed {{ box-shadow: inset 3px 0 0 var(--reviewed); }}
    .fn:hover {{ border-color: var(--line); }}
    .fn.selected {{ border-color: var(--hot); background: rgba(232,195,106,0.15); color: var(--hot); }}
    .fn.caller {{ border-color: var(--caller); background: rgba(107,179,232,0.12); }}
    .fn.callee {{ border-color: var(--callee); background: rgba(224,160,74,0.14); }}
    .fn.dim {{ opacity: 0.28; }}
    .detail {{ padding: 1rem; min-height: 280px; }}
    .detail.empty {{ color: var(--muted); font-size: 0.92rem; }}
    .detail .name {{
      font-family: var(--mono);
      font-size: 1.05rem;
      font-weight: 500;
      color: var(--hot);
      word-break: break-all;
    }}
    .detail .meta {{
      font-family: var(--mono);
      font-size: 0.75rem;
      color: var(--muted);
      margin: 0.35rem 0 1rem;
    }}
    .badge {{
      display: inline-block;
      font-family: var(--mono);
      font-size: 0.68rem;
      padding: 0.15rem 0.45rem;
      border-radius: 3px;
      margin-right: 0.35rem;
    }}
    .badge.new {{ background: rgba(94,200,160,0.18); color: var(--new); }}
    .badge.reviewed {{ background: rgba(122,143,163,0.2); color: #b8c8d6; }}
    .sec-title {{
      font-size: 0.72rem;
      font-family: var(--mono);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      margin: 1rem 0 0.4rem;
    }}
    .sec-title.calls {{ color: var(--callee); }}
    .sec-title.called {{ color: var(--caller); }}
    .link-list {{ list-style: none; margin: 0; padding: 0; }}
    .link-list li {{
      font-family: var(--mono);
      font-size: 0.8rem;
      padding: 0.35rem 0.5rem;
      border-radius: 4px;
      cursor: pointer;
      color: var(--text);
    }}
    .link-list li:hover {{ background: var(--bg-card); }}
    .link-list li .arrow {{ color: var(--muted); margin: 0 0.35rem; }}
    .flow-box {{
      margin-top: 1.25rem;
      background: var(--bg-elev);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 1rem;
    }}
    .flow-box h2 {{ margin: 0 0 0.75rem; font-size: 1rem; font-weight: 600; }}
    .flow-box p {{ margin: 0 0 0.75rem; color: var(--muted); font-size: 0.88rem; }}
    .entry-btns {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.75rem; }}
    .entry-btns button {{
      font-family: var(--mono);
      font-size: 0.75rem;
      padding: 0.4rem 0.7rem;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: var(--bg-card);
      color: var(--text);
      cursor: pointer;
    }}
    .entry-btns button:hover, .entry-btns button.on {{
      border-color: var(--accent-dim);
      background: rgba(61,184,138,0.12);
      color: var(--accent);
    }}
    #tree {{
      font-family: var(--mono);
      font-size: 0.78rem;
      line-height: 1.55;
      overflow-x: auto;
      white-space: pre;
      color: var(--text);
      background: #121820;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0.85rem 1rem;
      max-height: 420px;
      overflow-y: auto;
    }}
    #tree .t-new {{ color: var(--new); }}
    #tree .t-rev {{ color: var(--reviewed); }}
    #tree .t-sel {{ color: var(--hot); font-weight: 500; }}
    #tree .clickable {{ cursor: pointer; }}
    #tree .clickable:hover {{ text-decoration: underline; }}
    .legend {{
      display: flex; flex-wrap: wrap; gap: 1rem;
      font-size: 0.8rem; color: var(--muted);
      margin-top: 0.75rem;
    }}
    .legend i {{
      display: inline-block; width: 10px; height: 10px;
      border-radius: 2px; margin-right: 0.35rem; vertical-align: middle;
    }}
    .note {{
      margin-top: 1.5rem;
      font-size: 0.82rem;
      color: var(--muted);
      border-top: 1px solid var(--line);
      padding-top: 1rem;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <p class="eyebrow">querulus / integration</p>
      <h1>Карта функций</h1>
      <p>
        Кто кого вызывает внутри <code>service</code>, <code>scoring</code>, <code>features</code>.
        Клик по функции — список вызывающих и вызываемых.
        Маркеры <code style="color:var(--new)">#new</code> /
        <code style="color:var(--reviewed)">#reviewed</code>.
        <a href="architecture.html">← архитектура</a>
      </p>
    </header>

    <div class="toolbar">
      <input type="search" id="q" placeholder="поиск: score_second, FE, calibrator…" autocomplete="off" />
      <button type="button" class="chip tag-new on" data-tag="new">#new</button>
      <button type="button" class="chip tag-reviewed on" data-tag="reviewed">#reviewed</button>
      <span class="stats" id="stats"></span>
    </div>

    <div class="layout">
      <div class="panel">
        <div class="panel-h">Модули <span id="count-label"></span></div>
        <div id="modules"></div>
      </div>
      <div class="panel">
        <div class="panel-h">Связи <span>click → callers / callees</span></div>
        <div class="detail empty" id="detail">Выберите функцию слева или entry-point ниже.</div>
      </div>
    </div>

    <div class="flow-box">
      <h2>Дерево вызовов от entry-point</h2>
      <p>Исходящие вызовы (глубина ≤ 8). Повтор функции — ↻.</p>
      <div class="entry-btns" id="entries"></div>
      <div id="tree">Выберите entry-point…</div>
      <div class="legend">
        <span><i style="background:var(--new)"></i>#new</span>
        <span><i style="background:var(--reviewed)"></i>#reviewed</span>
        <span><i style="background:var(--hot)"></i>выбранная</span>
        <span><i style="background:var(--caller)"></i>вызывает выбранную</span>
        <span><i style="background:var(--callee)"></i>вызывается из выбранной</span>
      </div>
    </div>

    <p class="note">
      Граф собран статически (AST): прямые вызовы по имени внутри integration (без tests).
      Одинаковые имена в разных модулях резолвятся в модуль вызывающего.
      Внешние библиотеки не показаны. Пересобрать:
      <code style="font-family:var(--mono);color:var(--accent)">python examples/querulus/scripts/_gen_functions_html.py</code>
    </p>
  </div>

  <script>
  const DATA = {data_js};

  const byId = Object.fromEntries(DATA.nodes.map(n => [n.id, n]));
  const out = {{}};
  const inn = {{}};
  DATA.nodes.forEach(n => {{ out[n.id] = []; inn[n.id] = []; }});
  DATA.edges.forEach(e => {{
    if (out[e.from] && inn[e.to]) {{
      out[e.from].push(e.to);
      inn[e.to].push(e.from);
    }}
  }});
  Object.keys(out).forEach(k => {{ out[k] = [...new Set(out[k])].sort(); }});
  Object.keys(inn).forEach(k => {{ inn[k] = [...new Set(inn[k])].sort(); }});

  const ENTRIES = [
    'service.main.predict_route',
    'service.main.main_predict',
    'scoring.second.score_second',
    'scoring.second.preprocess_second_vector',
    'features.derived.add_second_model_features',
    'service.main.preprocessing_and_validate_vector',
    'service.main.health_route',
  ].filter(id => byId[id]);

  let selected = null;
  let entry = ENTRIES[0] || null;
  let tags = new Set(['new', 'reviewed']);
  let query = '';

  const $modules = document.getElementById('modules');
  const $detail = document.getElementById('detail');
  const $tree = document.getElementById('tree');
  const $stats = document.getElementById('stats');
  const $count = document.getElementById('count-label');
  const $entries = document.getElementById('entries');

  function short(id) {{
    const n = byId[id];
    return n ? n.name : id;
  }}

  function matchFilter(n) {{
    if (!tags.has(n.tag)) return false;
    if (!query) return true;
    const q = query.toLowerCase();
    return n.name.toLowerCase().includes(q)
      || n.module.toLowerCase().includes(q)
      || n.id.toLowerCase().includes(q);
  }}

  function renderModules() {{
    const groups = {{}};
    DATA.nodes.forEach(n => {{
      (groups[n.module] ||= []).push(n);
    }});
    const callers = selected ? new Set(inn[selected] || []) : new Set();
    const callees = selected ? new Set(out[selected] || []) : new Set();
    let visible = 0;
    $modules.innerHTML = Object.keys(groups).sort().map(mod => {{
      const fns = groups[mod].filter(matchFilter);
      visible += fns.length;
      if (!fns.length) return '';
      const secondish = mod.startsWith('scoring.second') || mod.startsWith('features');
      return `<div class="mod">
        <div class="mod-h${{secondish ? ' second' : ''}}"><span>${{mod}}</span><span>${{fns.length}}</span></div>
        <div class="mod-body">${{fns.map(n => {{
          let cls = 'fn ' + n.tag;
          if (selected === n.id) cls += ' selected';
          else if (selected) {{
            if (callers.has(n.id)) cls += ' caller';
            else if (callees.has(n.id)) cls += ' callee';
            else cls += ' dim';
          }}
          return `<button type="button" class="${{cls}}" data-id="${{n.id}}" title="${{n.id}}">${{n.name}}</button>`;
        }}).join('')}}</div>
      </div>`;
    }}).join('');
    $count.textContent = visible + ' / ' + DATA.nodes.length;
    $stats.textContent = DATA.nodes.length + ' fn · ' + DATA.edges.length + ' edges';
    $modules.querySelectorAll('.fn').forEach(btn => {{
      btn.addEventListener('click', () => select(btn.dataset.id));
    }});
  }}

  function renderDetail() {{
    if (!selected || !byId[selected]) {{
      $detail.className = 'detail empty';
      $detail.textContent = 'Выберите функцию слева или entry-point ниже.';
      return;
    }}
    const n = byId[selected];
    const calls = out[selected] || [];
    const called = inn[selected] || [];
    $detail.className = 'detail';
    $detail.innerHTML = `
      <div><span class="badge ${{n.tag}}">#${{n.tag}}</span></div>
      <div class="name">${{n.name}}</div>
      <div class="meta">${{n.file}}:${{n.lineno}} · ${{n.module}}</div>
      <div class="sec-title calls">Вызывает (${{calls.length}})</div>
      <ul class="link-list">${{calls.length ? calls.map(id =>
        `<li data-id="${{id}}"><span class="arrow">→</span>${{short(id)}} <span style="color:var(--muted)">(${{byId[id].module}})</span></li>`
      ).join('') : '<li style="color:var(--muted);cursor:default">—</li>'}}</ul>
      <div class="sec-title called">Вызывается из (${{called.length}})</div>
      <ul class="link-list">${{called.length ? called.map(id =>
        `<li data-id="${{id}}"><span class="arrow">←</span>${{short(id)}} <span style="color:var(--muted)">(${{byId[id].module}})</span></li>`
      ).join('') : '<li style="color:var(--muted);cursor:default">— (entry / leaf)</li>'}}</ul>
    `;
    $detail.querySelectorAll('.link-list li[data-id]').forEach(li => {{
      li.addEventListener('click', () => select(li.dataset.id));
    }});
  }}

  function renderTree() {{
    $entries.innerHTML = ENTRIES.map(id =>
      `<button type="button" class="${{entry===id?'on':''}}" data-id="${{id}}">${{short(id)}}</button>`
    ).join('');
    $entries.querySelectorAll('button').forEach(b => {{
      b.addEventListener('click', () => {{ entry = b.dataset.id; select(entry); }});
    }});
    if (!entry) {{ $tree.textContent = 'нет entry-points'; return; }}
    const lines = [];
    const seen = new Set();
    function walk(id, prefix, isLast, depth) {{
      if (depth > 8) {{
        lines.push(prefix + (isLast ? '└─ ' : '├─ ') + short(id) + ' …');
        return;
      }}
      const n = byId[id];
      if (!n) return;
      const branch = depth === 0 ? '' : (isLast ? '└─ ' : '├─ ');
      const mark = seen.has(id) ? ' ↻' : '';
      const cls = id === selected ? 't-sel' : (n.tag === 'new' ? 't-new' : 't-rev');
      lines.push(
        `${{prefix}}${{branch}}<span class="clickable ${{cls}}" data-id="${{id}}">${{n.name}}</span>` +
        `<span style="color:var(--muted)">  #${{n.tag}}${{mark}}</span>`
      );
      if (seen.has(id)) return;
      seen.add(id);
      const kids = out[id] || [];
      const nextPrefix = depth === 0 ? '' : prefix + (isLast ? '   ' : '│  ');
      kids.forEach((kid, i) => walk(kid, nextPrefix, i === kids.length - 1, depth + 1));
    }}
    walk(entry, '', true, 0);
    $tree.innerHTML = lines.join('\\n');
    $tree.querySelectorAll('.clickable').forEach(el => {{
      el.addEventListener('click', () => select(el.dataset.id));
    }});
  }}

  function select(id) {{
    selected = id;
    renderModules();
    renderDetail();
    renderTree();
  }}

  document.getElementById('q').addEventListener('input', e => {{
    query = e.target.value.trim();
    renderModules();
  }});
  document.querySelectorAll('.chip[data-tag]').forEach(chip => {{
    chip.addEventListener('click', () => {{
      const t = chip.dataset.tag;
      if (tags.has(t) && tags.size === 1) return;
      if (tags.has(t)) {{ tags.delete(t); chip.classList.remove('on'); }}
      else {{ tags.add(t); chip.classList.add('on'); }}
      renderModules();
    }});
  }});

  renderModules();
  renderTree();
  if (entry) select(entry);
  </script>
</body>
</html>
"""
    out = ROOT / "functions.html"
    out.write_text(html_doc, encoding="utf-8")
    print(f"wrote {out} ({len(nodes)} fn, {len(edges)} edges)")


if __name__ == "__main__":
    main()
