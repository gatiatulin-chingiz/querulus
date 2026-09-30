# Локальные команды Querulus.
.PHONY: synthetic-data test regenerate-configs

# Синтетический df_final для локального smoke (parquet в gitignore, в git не коммитить).
synthetic-data:
	python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path('src').resolve())); from querulus.synthetic_dataset import main; main()"

# Тесты правил features[]/DQ (схлопывание категорий, clip, NaN-политика).
test:
	python -m unittest discover -s tests -t . -v

# Перегенерация config_parity.json / config_prod.json из собранного parquet.
# В прод-контуре то же делает collect (ячейка export); здесь — локальный smoke.
regenerate-configs:
	python scripts/regenerate_outboxml_configs.py
