#!/bin/bash

exec python3.13 main.py \
  --prod_models_path "/home/model_user/list_models/" \
  --url_one "http://localhost:8080" \
  --host "0.0.0.0" \
  --port 8080
