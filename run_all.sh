#!/bin/bash

###### KILL CURRENT REST API SERVER
kill -9 `ps aux | grep app.js | grep -v grep | awk '{print $2}'`
netstat -lntp | grep 8080 | awk '{print $7}' | awk -F '/' '{print $1}' | xargs kill -9
###### START REST API SERVER
nohup npm run dev &

###### ACTIVATE PYTHON ENVIRONMENT FOR FLASK AI SERVER
activate () {
    . /home/ubuntu/2021AiHub-ODQA/python/.venv_v3/bin/activate
}
export FLASK_APP="/home/ubuntu/2021AiHub-ODQA/python/app.py"
activate
###### KILL CURRENT FLASK AI SERVER
netstat -lntp | grep 5000 | awk '{print $7}' | awk -F '/' '{print $1}' | xargs kill -9
###### START REST API SERVER
nohup flask run --host=127.0.0.1 --port=5000 &