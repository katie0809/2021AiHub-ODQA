#!/bin/bash
# activate () {
#     . /home/ubuntu/2021AiHub-ODQA/python/.venv_v2/bin/activate
# }
export FLASK_APP="app2.py"  # 리눅스 일때 <-
# activate
netstat -lntp | grep 8888 | awk '{print $7}' | awk -F '/' '{print $1}' | xargs kill -9
nohup flask run --host=0.0.0.0 --port=8888 &