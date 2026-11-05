"""Generate integration/functions_map.html call map from AST."""
from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from pathlib import Path

INTEGRATION = Path(__file__).resolve().parents[1] / "integration"
ROOT = INTEGRATION
OUT_PATH = INTEGRATION / "functions_map.html"
SKIP = {"__init__.py", "config.py"}
AMBIGUOUS = {"prepare_matrix", "require_columns", "_round_numbers"}

ENTRIES = [
    "main.predict_route",
    "main.main_predict",
    "main.lifespan",
    "main._startup",
    "src.predictions.score_main",
    "src.predictions.score_second",
    "src.predictions.preload_models",
    "src.preprocessing.second.preprocess.preprocess_second_vector",
    "src.preprocessing.common.preprocess_input_vector",
    "src.preprocessing.common.coerce_and_enrich",
    "src.preprocessing.common.prepare_vector_upper",
    "src.preprocessing.second.features.add_second_model_features",
    "main.health_route",
]

# Короткие описания для карты (человеческий язык). Ключ — имя функции.
HUMAN: dict[str, str] = {
    "predict_route": "Принимает HTTP-запрос на скоринг и отдаёт ответ",
    "main_predict": "Считает основной трек и при необходимости второй",
    "lifespan": "При старте сервиса поднимает модели в память",
    "_startup": "Пишет лог, грузит модели и гоняет контрольный пример",
    "health_route": "Отвечает, что сервис жив, и какие версии треков",
    "score_main": "Полный скоринг основной модели: от входа до ответа",
    "score_second": "Полный скоринг второй модели в том же формате ответа",
    "preload_models": "Заранее загружает модели и калибраторы в память",
    "_load_model_pair": "Достаёт из файла пару моделей классификации и регрессии",
    "_classification_proba_and_labels": "Считает вероятности и метки 0/1 по порогу",
    "_raw_classification_scores": "Сырой скор классификации до калибровки",
    "_round_numbers": "Округляет числа в ответе до двух знаков",
    "run_classification_model": "Прогоняет классификацию и возвращает метки",
    "run_regression_model": "Прогоняет регрессию и возвращает суммы",
    "preprocess_second_vector": "Готовит вход и считает фичи для второй модели",
    "preprocess_input_vector": "Общий вход main_/second_: UPPER → касты → validate",
    "coerce_and_enrich": "Приводит типы, даты, форму заявителя и NOT_NOTIFICATION к нужному виду",
    "prepare_vector_upper": "Собирает таблицу из JSON и поднимает строки в UPPER",
    "prepare_matrix": "Подгоняет таблицу под то, что ждала модель при обучении",
    "require_columns": "Проверяет наличие колонок и режет кадр в нужном порядке",
    "resolve_threshold": "Берёт порог классификации из запроса или из настроек",
    "df_to_records": "Переводит таблицу в список словарей для JSON",
    "validate_input_vector": "Проверяет полный входной контракт и preprocess-инварианты",
    "validate_model_features": "После FE проверяет только engineered-колонки",
    "validate_preprocessed_vector": "UPPER / дубли колонок / даты после enrich",
    "_amount_bins": "Режет сумму на три денежных бакета",
    "validate_derived_dates_strict": "Проверяет, что даты после подготовки выглядят корректно",
    "validate_no_duplicate_columns": "Запрещает повторяющиеся имена колонок",
    "add_second_model_features": "Считает дополнительные признаки для второй модели",
    "_safe_div": "Делит числа и подставляет пустоту при делении на ноль",
    "add_real_monetary_columns": "Пересчитывает суммы в цены базового года",
    "deflate_to_base_year": "Убирает инфляцию из денежных сумм по готовому году",
    "resolve_cpi_year": "Подбирает год, по которому искать индекс цен",
    "real_feature_name": "Собирает имя колонки для суммы без инфляции",
    "apply_cf_calibrator": "Isotonic-калибровка вероятностей классификации",
    "apply_rg_calibrator": "Isotonic-калибровка сумм регрессии (clip ≥ 0)",
    "shap_top_features_comment": "Короткий текст, какие признаки сильнее всего повлияли",
    "unwrap_mldataworker_model": "Снимает служебные обёртки вокруг модели",
    "_catboost_shap_values": "SHAP через CatBoost get_feature_importance",
    "masked_predictions": "Накладывает бизнес-правила на суммы регрессии",
    "calibrator_paths": "Пути к файлам калибраторов второго трека",
    "feature_names_from_model_result": "Читает список признаков из сохранённой модели",
    "model_pickle_path": "Строит путь к файлу модели",
    "prod_models_dir": "Папка, где лежат модели и калибраторы",
    "assert_startup_scores": "Прогоняет контрольный пример при старте сервиса",
    "_check_result": "Проверяет результат контрольного скоринга",
    "install_mldataworker_shim": "Подставляет заглушку, если нет настоящего mldataworker",
    "_alias_models": "Подменяет модуль моделей на outboxml при необходимости",
    "_ensure_pkg": "Гарантирует, что пакет виден в sys.modules",
}


def _summary(node: ast.AST) -> str:
    """Человеческое описание: HUMAN или очищенный docstring."""
    name = getattr(node, "name", "") or ""
    if name in HUMAN:
        return HUMAN[name]
    doc = ast.get_docstring(node) or ""
    for line in doc.splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith(("Args:", "Returns:", "Raises:", "Шаги:")):
            break
        text = text.replace("``", "").replace("`", "")
        text = re.sub(r"\s+", " ", text)
        return text[:220]
    return ""


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
                if t in ("# new", "# changed", "# reviewed"):
                    tag = t[2:].strip()
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
                "summary": _summary(node),
            }
            calls = [sub for sub in ast.walk(node) if isinstance(sub, ast.Call)]
            calls.sort(
                key=lambda c: (
                    getattr(c, "lineno", 0) or 0,
                    getattr(c, "col_offset", 0) or 0,
                )
            )
            for sub in calls:
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

    resolved_order: list[tuple[str, str]] = []
    seen_edges: set[tuple[str, str]] = set()
    for caller, callee in edges_raw:
        cands = by_name.get(callee, [])
        if not cands:
            continue
        cmod = mods[caller]["module"]
        same = [c for c in cands if mods[c]["module"] == cmod]
        # Свой модуль предпочтительнее; если имени нет локально —
        # берём внешние кандидаты (кроме по-настоящему неоднозначных).
        if same:
            targets = same
        elif callee in AMBIGUOUS and len(cands) > 1:
            continue
        else:
            targets = cands
        for t in targets:
            if t == caller:
                continue
            edge = (caller, t)
            if edge not in seen_edges:
                seen_edges.add(edge)
                resolved_order.append(edge)

    nodes = sorted(mods.values(), key=lambda x: (x["module"], x["name"]))
    edges = [{"from": a, "to": b} for a, b in resolved_order]
    return nodes, edges


def main() -> None:
    nodes, edges = collect()
    data_js = json.dumps(
        {"nodes": nodes, "edges": edges, "entries": ENTRIES},
        ensure_ascii=False,
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Querulus Integration — дерево функций</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet" />
  <style>
    :root {{
      --bg: #10141a;
      --panel: #171d26;
      --card: #1c2430;
      --line: #2e3846;
      --text: #e8eef4;
      --muted: #8fa0b3;
      --new: #4a9eff;
      --changed: #f0c040;
      --reviewed: #3ecf8e;
      --unknown: #8fa0b3;
      --arrow: #7a8fa3;
      --mono: "IBM Plex Mono", ui-monospace, monospace;
      --sans: "IBM Plex Sans", system-ui, sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: var(--sans);
      background: var(--bg);
      color: var(--text);
    }}
    .wrap {{ padding: 0.75rem 0.85rem 1rem; }}
    header .eyebrow {{
      font-family: var(--mono);
      font-size: 0.68rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin: 0 0 0.2rem;
    }}
    h1 {{ margin: 0 0 0.25rem; font-size: 1.2rem; }}
    header p {{ margin: 0; color: var(--muted); font-size: 0.82rem; }}
    .legend {{
      display: flex; flex-wrap: wrap; gap: 0.85rem;
      margin: 0.5rem 0; font-family: var(--mono); font-size: 0.72rem;
    }}
    .legend span {{ display: inline-flex; align-items: center; gap: 0.3rem; }}
    .legend i {{ width: 8px; height: 8px; border-radius: 2px; display: inline-block; }}
    .c-new {{ background: var(--new); }}
    .c-changed {{ background: var(--changed); }}
    .c-reviewed {{ background: var(--reviewed); }}
    .c-unknown {{ background: var(--unknown); }}
    .toolbar {{
      display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center;
      margin-bottom: 0.4rem;
    }}
    .toolbar input {{
      flex: 1 1 200px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 5px;
      color: var(--text);
      padding: 0.35rem 0.55rem;
      font-family: var(--mono);
      font-size: 0.78rem;
    }}
    .entry-btns {{ display: flex; flex-wrap: wrap; gap: 0.28rem; margin-bottom: 0.5rem; }}
    .entry-btns button {{
      font-family: var(--mono); font-size: 0.68rem;
      padding: 0.22rem 0.45rem; border-radius: 4px;
      border: 1px solid var(--line); background: var(--panel);
      color: var(--muted); cursor: pointer;
    }}
    .entry-btns button.on {{
      color: var(--text); border-color: #4a6a8a; background: #1c2836;
    }}
    .stats {{ margin-left: auto; font-family: var(--mono); font-size: 0.68rem; color: var(--muted); }}
    .tree-box {{
      position: relative;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: auto;
      width: 100%;
      height: calc(100vh - 160px);
    }}
    #wires {{
      position: absolute;
      left: 0; top: 0;
      pointer-events: none;
      overflow: visible;
      z-index: 0;
    }}
    .vtree {{
      position: relative;
      z-index: 1;
      display: flex;
      align-items: flex-start;
      min-width: max-content;
      padding: 1.25rem 1.5rem 2rem;
    }}
    /* unit = карточка + дети справа */
    .unit {{
      display: flex;
      flex-direction: row;
      align-items: flex-start;
      gap: 1.35rem;
    }}
    .unit.spine {{
      align-items: flex-start;
    }}
    .unit.spine > .card {{
      margin-top: 0.15rem;
    }}
    .card {{
      position: relative;
      flex: 0 0 auto;
      max-width: 280px;
      background: var(--card);
      border: 1px solid var(--line);
      border-left: 3px solid var(--unknown);
      border-radius: 5px;
      padding: 0.3rem 1.15rem 0.35rem 0.45rem;
    }}
    .card.new {{ border-left-color: var(--new); }}
    .card.changed {{ border-left-color: var(--changed); }}
    .card.reviewed {{ border-left-color: var(--reviewed); }}
    .card .name {{
      font-family: var(--mono);
      font-size: 0.74rem;
      font-weight: 600;
      display: block;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .card.new .name {{ color: var(--new); }}
    .card.changed .name {{ color: var(--changed); }}
    .card.reviewed .name {{ color: var(--reviewed); }}
    .card.unknown .name {{ color: var(--unknown); }}
    .card .meta {{
      margin-top: 0.2rem;
      font-size: 0.68rem;
      line-height: 1.25;
      color: var(--muted);
      white-space: normal;
    }}
    .card .toggle {{
      position: absolute;
      top: 0.2rem; right: 0.2rem;
      width: 0.95rem; height: 0.95rem;
      border: 1px solid var(--line);
      background: #141a22; color: var(--muted);
      border-radius: 3px; cursor: pointer;
      font-size: 0.62rem; line-height: 1; padding: 0;
    }}
    /* depth 0–1 и depth≥3: дети в ряд справа */
    .kids-row {{
      display: flex;
      flex-direction: row;
      align-items: flex-start;
      gap: 1.1rem;
    }}
    /* depth 2 (score_*): дети столбиком справа */
    .kids-col {{
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      gap: 0.75rem;
    }}
    .card.priv {{ opacity: 0.9; }}
    .loop {{ font-family: var(--mono); font-size: 0.62rem; color: var(--muted); }}
    .note {{ margin-top: 0.5rem; font-size: 0.72rem; color: var(--muted); }}
    .note code {{ font-family: var(--mono); color: var(--new); }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <p class="eyebrow">querulus / integration</p>
      <h1>Дерево вызовов</h1>
      <p>
        Один вызов — цепочка вправо; несколько — ветви столбиком.
        <code>score_second</code> под <code>score_main</code>.
      </p>
    </header>
    <div class="legend">
      <span><i class="c-new"></i># new</span>
      <span><i class="c-changed"></i># changed</span>
      <span><i class="c-reviewed"></i># reviewed</span>
      <span><i class="c-unknown"></i># unknown</span>
    </div>
    <div class="toolbar">
      <input type="search" id="q" placeholder="фильтр…" autocomplete="off" />
      <span class="stats" id="stats"></span>
    </div>
    <div class="entry-btns" id="entries"></div>
    <div class="tree-box" id="box">
      <svg id="wires" xmlns="http://www.w3.org/2000/svg"></svg>
      <div class="vtree" id="tree"></div>
    </div>
    <p class="note">
      Пересобрать: <code>python examples/querulus/scripts/_gen_functions_html.py</code>
    </p>
  </div>
  <script>
  const DATA = {data_js};
  const byId = Object.fromEntries(DATA.nodes.map(n => [n.id, n]));
  const out = {{}};
  DATA.nodes.forEach(n => {{ out[n.id] = []; }});
  DATA.edges.forEach(e => {{
    if (out[e.from] && !out[e.from].includes(e.to)) out[e.from].push(e.to);
  }});

  const ENTRIES = (DATA.entries || []).filter(id => byId[id]);
  let entry = ENTRIES[0] || null;
  let query = '';
  const collapsed = new Set();
  const $box = document.getElementById('box');
  const $tree = document.getElementById('tree');
  const $wires = document.getElementById('wires');
  const $entries = document.getElementById('entries');
  const $stats = document.getElementById('stats');

  function esc(s) {{
    return String(s || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }}

  function matchNode(n) {{
    if (!query) return true;
    const q = query.toLowerCase();
    return n.name.toLowerCase().includes(q)
      || n.module.toLowerCase().includes(q)
      || (n.summary || '').toLowerCase().includes(q)
      || n.id.toLowerCase().includes(q);
  }}

  function subtreeMatches(id, stack) {{
    const n = byId[id];
    if (!n) return false;
    if (matchNode(n)) return true;
    if (stack.has(id)) return false;
    const next = new Set(stack);
    next.add(id);
    return (out[id] || []).some(cid => subtreeMatches(cid, next));
  }}

  function reaches(fromId, targetId, seen) {{
    if (fromId === targetId) return true;
    if (seen.has(fromId)) return false;
    seen.add(fromId);
    return (out[fromId] || []).some(cid => reaches(cid, targetId, seen));
  }}

  function visibleKids(id) {{
    const kids = out[id] || [];
    const kept = [];
    kids.forEach(kid => {{
      if (!kept.some(other => reaches(other, kid, new Set([id])))) kept.push(kid);
    }});
    return kept;
  }}

  const SPINE = new Set(['predict_route', 'main_predict', 'score_main', 'score_second']);
  // 1 ребёнок → цепочка вправо; 2+ → ветви столбиком (и score_*/main_predict всегда столбик).
  function kidsClass(name, count) {{
    if (name === 'main_predict' || name === 'score_main' || name === 'score_second') {{
      return 'kids-col';
    }}
    return count >= 2 ? 'kids-col' : 'kids-row';
  }}

  function cardHtml(n, looping, toggle, nid) {{
    const tip = `#${{n.tag}} · ${{n.file}}:${{n.lineno}}`;
    const desc = (n.summary || '').trim() || tip;
    const priv = n.name.startsWith('_') ? ' priv' : '';
    return `<div class="card ${{esc(n.tag)}}${{priv}}" data-nid="${{esc(nid)}}" title="${{esc(tip)}}">
      ${{toggle || ''}}
      <span class="name">${{esc(n.name)}}${{looping ? ' <span class="loop">↻</span>' : ''}}</span>
      <div class="meta">${{esc(desc)}}</div>
    </div>`;
  }}

  function renderNode(id, stack) {{
    const n = byId[id];
    if (!n) return '';
    if (query && !subtreeMatches(id, new Set(stack))) return '';

    const kids = visibleKids(id);
    const looping = stack.has(id);
    const key = [...stack].join('>') + '>' + id;
    const isCollapsed = collapsed.has(key);
    const hasKids = kids.length > 0 && !looping;
    const toggle = hasKids
      ? `<button type="button" class="toggle" data-key="${{esc(key)}}">${{isCollapsed ? '+' : '−'}}</button>`
      : '';

    let kidsHtml = '';
    if (hasKids && !isCollapsed) {{
      const next = new Set(stack);
      next.add(id);
      const items = kids.map(cid => renderNode(cid, next)).filter(Boolean);
      if (items.length) {{
        kidsHtml = `<div class="${{kidsClass(n.name, items.length)}}" data-from="${{esc(id)}}">${{items.join('')}}</div>`;
      }}
    }}

    const spine = SPINE.has(n.name) ? ' spine' : '';
    return `<div class="unit${{spine}}">${{cardHtml(n, looping, toggle, id)}}${{kidsHtml}}</div>`;
  }}

  function drawWires() {{
    const boxRect = $box.getBoundingClientRect();
    const scrollX = $box.scrollLeft;
    const scrollY = $box.scrollTop;
    const width = Math.max($box.clientWidth, $tree.scrollWidth + 40);
    const height = Math.max($box.clientHeight, $tree.scrollHeight + 40);
    $wires.setAttribute('width', width);
    $wires.setAttribute('height', height);
    $wires.style.width = width + 'px';
    $wires.style.height = height + 'px';

    const pairs = [];
    $tree.querySelectorAll('.kids-row[data-from], .kids-col[data-from]').forEach(group => {{
      const parentCard = group.parentElement
        && group.parentElement.querySelector(':scope > .card[data-nid]');
      if (!parentCard) return;
      // прямые дети: .unit > .card (включая leaf)
      group.querySelectorAll(':scope > .unit > .card[data-nid]').forEach(childCard => {{
        pairs.push([parentCard, childCard]);
      }});
    }});

    const ns = 'http://www.w3.org/2000/svg';
    while ($wires.firstChild) $wires.removeChild($wires.firstChild);
    const marker = document.createElementNS(ns, 'defs');
    marker.innerHTML = `<marker id="arrowHead" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#7a8fa3" />
    </marker>`;
    $wires.appendChild(marker);

    pairs.forEach(([parentCard, childCard]) => {{
      const pr = parentCard.getBoundingClientRect();
      const cr = childCard.getBoundingClientRect();
      // всегда справа от родителя: правый край → левый край ребёнка
      const x1 = pr.right - boxRect.left + scrollX;
      const y1 = pr.top - boxRect.top + scrollY + pr.height / 2;
      const x2 = cr.left - boxRect.left + scrollX;
      const y2 = cr.top - boxRect.top + scrollY + cr.height / 2;
      const midX = x1 + Math.max(12, (x2 - x1) / 2);
      const path = document.createElementNS(ns, 'path');
      path.setAttribute('d', `M ${{x1}} ${{y1}} H ${{midX}} V ${{y2}} H ${{x2 - 4}}`);
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', '#7a8fa3');
      path.setAttribute('stroke-width', '1.5');
      path.setAttribute('marker-end', 'url(#arrowHead)');
      $wires.appendChild(path);
    }});
  }}

  function renderEntries() {{
    $entries.innerHTML = ENTRIES.map(id =>
      `<button type="button" class="${{entry === id ? 'on' : ''}}" data-id="${{esc(id)}}">${{esc(byId[id].name)}}</button>`
    ).join('');
    $entries.querySelectorAll('button').forEach(btn => {{
      btn.addEventListener('click', () => {{
        entry = btn.dataset.id;
        collapsed.clear();
        render();
      }});
    }});
  }}

  function render() {{
    renderEntries();
    if (!entry) {{
      $tree.innerHTML = '<div style="color:var(--muted)">нет entry-point</div>';
      $wires.innerHTML = '';
      return;
    }}
    $tree.innerHTML = renderNode(entry, new Set());
    $stats.textContent = DATA.nodes.length + ' fn · ' + DATA.edges.length + ' edges';
    $tree.querySelectorAll('.toggle[data-key]').forEach(el => {{
      el.addEventListener('click', () => {{
        const key = el.dataset.key;
        if (collapsed.has(key)) collapsed.delete(key);
        else collapsed.add(key);
        render();
      }});
    }});
    requestAnimationFrame(() => requestAnimationFrame(drawWires));
  }}

  document.getElementById('q').addEventListener('input', e => {{
    query = e.target.value.trim();
    render();
  }});
  $box.addEventListener('scroll', () => drawWires());
  window.addEventListener('resize', () => drawWires());

  render();
  </script>
</body>
</html>
"""
    OUT_PATH.write_text(html_doc, encoding="utf-8")
    print(f"wrote {OUT_PATH} ({len(nodes)} fn, {len(edges)} edges)")


if __name__ == "__main__":
    main()
