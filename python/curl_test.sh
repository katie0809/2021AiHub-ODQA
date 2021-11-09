 curl http://localhost:5000/predict -X POST -H "Content-Type: application/json; charset=utf-8" \
-d " \
{ \
 \"question\":\"고속철도차량의 소음을 저감하기 위해서는 무엇이 중요한가?\", \
 \"context\": \"\n마이크로폰 어레이를 이용한 고속철도 차량의 소음원 도출 연구\"\
} \
" 
