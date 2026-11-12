#!/bin/bash

exec python3.13 main.py \
  --prod_models_path "/home/model_user/list_models/" \
  --host "0.0.0.0" \
  --port 8080
