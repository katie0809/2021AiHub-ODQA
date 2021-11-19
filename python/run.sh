#!/bin/bash
export FLASK_APP="predict_with_contextlist.py"  # 리눅스 일때 <-
flask run --host=127.0.0.1 --port=5000 &
flask run --host=127.0.0.1 --port=5001 &
flask run --host=127.0.0.1 --port=5002 &
# flask run --host=127.0.0.1 --port=5003 &
# flask run --host=127.0.0.1 --port=5004 &
# flask run --host=127.0.0.1 --port=5005 &
# flask run --host=127.0.0.1 --port=5006 &
# flask run --host=127.0.0.1 --port=5007 &
# flask run --host=127.0.0.1 --port=5008 &
# flask run --host=127.0.0.1 --port=5009 &
