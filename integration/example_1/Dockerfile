FROM repo.vsk.ru:5001/python:3.12

USER root
COPY pip.conf /etc/pip.conf
COPY req.txt /home/model_user/model/req.txt
RUN pip install -r /home/model_user/model/req.txt
ENV TZ=Europe/Moscow
RUN mv /etc/localtime /etc/localtime.bak
RUN ln -s /usr/share/zoneinfo/Europe/Moscow /etc/localtime

RUN useradd model_user
RUN mkdir -p /home/model_user/
RUN chown model_user:model_user /home/model_user/
USER model_user

RUN mkdir -p /home/model_user/list_models/

COPY --chown=model_user:model_user main.py /home/model_user/model/main.py
COPY --chown=model_user:model_user make_service.sh /home/model_user/model/make_service.sh

RUN chmod +x /home/model_user/model/make_service.sh

WORKDIR /home/model_user/model

EXPOSE 8080

ENTRYPOINT ["/home/model_user/model/make_service.sh"]
