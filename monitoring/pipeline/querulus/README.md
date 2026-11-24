# Contour vendor: `querulus.fin_effect`

На GitLab `querulus_monitoring.git` нет monorepo `src/`.
Сюда скопирован минимальный набор модулей для `make fin_effect`
(`monitoring_snapshots` / `excel_monitoring` / …).

Локально в monorepo предпочтителен `examples/querulus/src`
(`fin_effect.py` добавляет его в `sys.path` первым).

При изменении расчёта в `src/querulus/fin_effect/` — обновить копии здесь.
