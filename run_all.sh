#!/bin/bash
netstat -lntp | grep 8080 | awk '{print $7}' | awk -F '/' '{print $1}' | xargs kill -9
#npm run dev
nohup npm run dev &
