import json
from pathlib import Path

nb_path = Path(__file__).parent / "example_2.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))

for cell in nb["cells"]:
    lines = cell.get("source", [])
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line == "from pathlib import Path\n" and "QUERULUS_DIR" not in "".join(lines):
            out.extend([
                "from pathlib import Path\n",
                "\n",
                "QUERULUS_DIR = Path(__file__).resolve().parent\n",
                "DATA_DIR = QUERULUS_DIR.parent / 'data'\n",
                "DATA_PROCESSED = DATA_DIR / 'processed'\n",
                "CONFIG_DIR = QUERULUS_DIR / 'configs'\n",
                "CONFIG_CF = CONFIG_DIR / 'config_cf_3.json'\n",
                "\n",
            ])
            i += 1
            continue
        if "DATA_DIR = Path('data')" in line:
            i += 1
            continue
        line = line.replace("config_name='.configs/", "config_name='./configs/")
        line = line.replace("config_name='configs./", "config_name='./configs/")
        line = line.replace(
            "ds_manager_cl = DataSetsManager(config_name='./configs/config_cf_3.json'",
            "ds_manager_cl = DataSetsManager(config_name=str(CONFIG_CF)",
        )
        line = line.replace(
            "ds_manager_cl = DataSetsManager(config_name='./config_cf_3.json'",
            "ds_manager_cl = DataSetsManager(config_name=str(CONFIG_CF)",
        )
        out.append(line)
        if line.strip() == "ds_manager_cl.load_dataset(data=data_cl)":
            out.append("ds_manager_cl._data_preprocessor._dataset = ds_manager_cl._extractor\n")
        if "ds_manager_rg.load_dataset(data=data_rg)" in line and "_data_preprocessor" not in line:
            out.append("ds_manager_rg._data_preprocessor._dataset = ds_manager_rg._extractor\n")
        if "CONFIG_RG" not in "".join(out) and "TEST_RG_PATH" in line:
            idx = out.index(line)
            out.insert(idx, "CONFIG_RG = CONFIG_DIR / 'config_rg_3.json'\n")
        if "ds_manager_rg = DataSetsManager" in line and "CONFIG_RG" in line.replace("config_rg_3", ""):
            pass
        if "ds_manager_rg = DataSetsManager(config_name='./configs/config_rg_3.json'" in line:
            out[-1] = "ds_manager_rg = DataSetsManager(config_name=str(CONFIG_RG), external_config=config)\n"
        i += 1
    cell["source"] = out

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok")
