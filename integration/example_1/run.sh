#!/bin/bash

# Установим значение по умолчанию для URL
URL_ONE="http://localhost:8080"

# Проверка значения переменной среды profile
if [ "$SPRING_PROFILE" == "test" ]; then
  URL_ONE="http://perf-pydsp-test1.vsk.ru:10609"
fi
if [ "$SPRING_PROFILE" == "prod" ]; then
  URL_ONE="http://perf-pydsp-cls.vsk.ru:10609"
fi

# Запуск приложения с вычисленным URL
exec python3.12 main.py \
  --prod_models_path "/home/model_user/list_models/" \
  --url_one "$URL_ONE" \
  --host "0.0.0.0" \
  --port 8080
