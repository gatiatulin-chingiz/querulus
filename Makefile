# Локальные команды Querulus.
.PHONY: synthetic-data

# Синтетический df_final для локального smoke (parquet в gitignore).
synthetic-data:
	python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path('src').resolve())); from querulus.synthetic_dataset import main; main()"
