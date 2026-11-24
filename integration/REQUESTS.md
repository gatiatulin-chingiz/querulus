# Запросы к Querulus (bash + curl)

| Контур | `$BASE` |
|--------|---------|
| local (`python main.py`) | `http://localhost:8080` |
| local Docker | `http://localhost:10655` |
| test | `http://perf-pydsp-test-w1.vsk.ru:10655` |
| prod | `http://perf-pydsp-cls.vsk.ru:10655` |

Stem’ы: main `querulus_ansamble_2026_04_18_v1`, second `UU_querulus_2026_11_05_12_44_24`.  
Payload dual-track: `payload_model.json`.

```bash
# health
curl -sS "$BASE/api/health"

# predict one
curl -sS -X POST "$BASE/api/predict_one" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "querulus_ansamble_2026_04_18_v1",
    "request": [{
      "RECIEVE_METHOD": "Партнер",
      "APPLICANT_FORM": "Потерпевший",
      "APPLICANT_AGE": 39,
      "FILIAL": "Алтайский",
      "EVENT_CREATED_BY_GIBDD_FLAG": 1,
      "VICTIM_MAX_WEIGHT": 2110,
      "VICTIM_VEHICLE_AGE": 13,
      "LOSS_UNIT_ZONE": "Зона ф-ла Алтайский",
      "EVENT_DATE": "2022-11-05T07:15:00",
      "LOSS_DATE_TIME": "2022-11-06T00:00:00",
      "LOSS_NUMBER": 8940869,
      "AMOUNT_REPAIR": 242542,
      "VICTIM_VEHICLE_BRAND": "Toyota",
      "PAYMENT_ORDER_DATE_TIME": "2022-11-17T00:00:00",
      "VICTIM_VEHICLE_CATEGORY": "Средний 7+",
      "INCIDENT_NUMBER": 8940869,
      "VALUE_BEFORE_WITH": 134913,
      "FL_PHOTO_VIDEO": 0,
      "VALUE_BEFORE_WITHOUT": 242542,
      "PAYMENT_RECIPIENT_TYPE": "Физ. Лицо",
      "LOSS_UNIT_TYPE": "Основной офис",
      "VICTIM_OBJECT_YEAR": 2009,
      "VICTIM_AGE": 39,
      "IS_TOTAL": 0,
      "IS_LAWYER": 0,
      "WORK": 11128,
      "VICTIM_CAPACITY_ENGINE": 2,
      "POLICYHOLDER_CONTRACT_END_DATE": "2022-12-21T00:00:00",
      "NOT_NOTIFICATION": "SMS",
      "REGION": "Алтайский",
      "GUILTY_MAX_WEIGHT": 2110,
      "VICTIM_OBJECT_REG_NUM": "А708РВ122",
      "GUILTY_CAPACITY_ENGINE": 1.3,
      "VICTIM_VEHICLE_COUNTRY": "РОССИЯ"
    }],
    "submodel": null
  }'

# dual-track (из integration/)
curl -sS -X POST "$BASE/api/predict" \
  -H "Content-Type: application/json" \
  -d @payload_model.json
```

Локальный старт:

```bash
python main.py --prod_models_path ./results --host 0.0.0.0 --port 8080 --url_one http://localhost:8080
```
