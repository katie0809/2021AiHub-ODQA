#!/bin/bash
activate () {
    . /home/ubuntu/2021AiHub-ODQA/python/.venv_v3/bin/activate
}
export FLASK_APP="app.py"  # 리눅스 일때 <-
activate
netstat -lntp | grep 5000 | awk '{print $7}' | awk -F '/' '{print $1}' | xargs kill -9
nohup flask run --host=127.0.0.1 --port=5000 &