FROM repo.vsk.ru:5001/python:3.13

RUN useradd -m model_user

COPY pip.conf /etc/pip.conf
COPY requirements.txt /home/model_user/model/requirements.txt
RUN pip install --no-cache-dir --timeout=120 -r /home/model_user/model/requirements.txt

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN mkdir -p /home/model_user/list_models
RUN chown -R model_user:model_user /home/model_user

USER model_user

COPY --chown=model_user:model_user main.py /home/model_user/model/main.py
COPY --chown=model_user:model_user preprocessing.py /home/model_user/model/preprocessing.py
COPY --chown=model_user:model_user make_service.sh /home/model_user/model/make_service.sh

RUN chmod +x /home/model_user/model/make_service.sh

WORKDIR /home/model_user/model

EXPOSE 8080

ENTRYPOINT ["/home/model_user/model/make_service.sh"]
