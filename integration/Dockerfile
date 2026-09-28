FROM repo.vsk.ru:5001/python:3.13

# 1. Создаём пользователя с домашней директорией
RUN useradd -m model_user

# 2. Конфигурация pip и установка зависимостей (от root)
COPY pip.conf /etc/pip.conf
COPY requirements.txt /home/model_user/model/requirements.txt
RUN pip install --no-cache-dir --timeout=120 -r /home/model_user/model/requirements.txt

# 3. Настройка часового пояса
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 4. Создаём рабочие директории и назначаем владельца
RUN mkdir -p /home/model_user/model/results /home/model_user/list_models
RUN chown -R model_user:model_user /home/model_user

# 5. Переключаемся на непривилегированного пользователя
USER model_user

# 6. Копируем модели и исходный код (legacy flat results/ + shadow querulus/2.0.0/)
COPY --chown=model_user:model_user results/ /home/model_user/model/results/
COPY --chown=model_user:model_user main.py /home/model_user/model/main.py
COPY --chown=model_user:model_user config.py /home/model_user/model/config.py
COPY --chown=model_user:model_user features_cpi.py /home/model_user/model/features_cpi.py
COPY --chown=model_user:model_user features_dq.py /home/model_user/model/features_dq.py
COPY --chown=model_user:model_user model_profiles.py /home/model_user/model/model_profiles.py
COPY --chown=model_user:model_user shadow_v2.py /home/model_user/model/shadow_v2.py

# 7. Рабочая директория и точка входа
WORKDIR /home/model_user/model

EXPOSE 8080
ENTRYPOINT ["python3.13", "main.py"]
