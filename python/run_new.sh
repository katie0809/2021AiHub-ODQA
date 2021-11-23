#!/bin/bash
activate () {
    . /home/ubuntu/2021AiHub-ODQA/python/.venv_v2/bin/activate
}
export FLASK_APP="app2.py"  # 리눅스 일때 <-
activate
netstat -lntp | grep 8888 | awk '{print $7}' | awk -F '/' '{print $1}' | xargs kill -9
nohup flask run --host=0.0.0.0 --port=8888 &
# nohup flask run --host=127.0.0.1 --port=5001 &
# nohup flask run --host=127.0.0.1 --port=5002 &
# nohup flask run --host=127.0.0.1 --port=5003 &
# flask run --host=127.0.0.1 --port=5004 &
# flask run --host=127.0.0.1 --port=5005 &
# flask run --host=127.0.0.1 --port=5006 &
# flask run --host=127.0.0.1 --port=5007 &
# flask run --host=127.0.0.1 --port=5008 &
# flask run --host=127.0.0.1 --port=5009 &
