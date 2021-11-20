const strings = require('../config/strings');
const logger = require('./Logger');
const createError = require('http-errors');
const axios = require('axios');
const FormData = require('form-data');

const instance = axios.create();
instance.defaults.timeout = 5000; //3sec

async function predict(context, question) {

	// 1. 여러서버에 분리하여 전송 후 받는 경우
	// let port_list = ["5000","5001"]
	// let call_list = []
	// let result = []
	// var step;
	// for (step = 0; step < context.length; step++) {
	
	// 	let payload = JSON.stringify({
	// 		question: question,
	// 		context: [context[step]]
	// 	});
	// 	let host = 'http://127.0.0.1:' + port_list[step%port_list.length] +'/predict'
	// 	logger.debug("predict server : ",host);

	// 	call_list.push(
	// 		instance.post(host, payload, 
	// 				{ headers: {"Content-Type": "application/json; charset=utf-8"}})
	// 			.then((value)=> {
	// 				let data = value.data;
	// 				logger.debug("data ",data);
	// 				result.push(data);
	// 			})
	// 	)
	// }
	// await Promise.all(call_list);
	// logger.debug("result", result);
	// await context.forEach(async element => {
		
	// 	let payload = JSON.stringify({
	// 		question: question[0],
	// 		context: [element]
	// 	});
	// 	let host = 'http://127.0.0.1:' + port_list[port_idx++] +'/predict'
	// 	console.log("predict server : ",host);
	// 	let res = await axios.post(host, payload, 
    // 	    { headers: {"Content-Type": "application/json; charset=utf-8"}});
	// 	//port_idx += 1;
	// 	port_idx = port_idx >= port_list.length ? 0 : port_idx;
	// 	let data = res.data;
	// 	console.log(data);
	// 	result.push[data];
	// });

	// 2. 1개 서버에 전체 전달 후 답변 받는 경우
	let payload = JSON.stringify({
				question: question,
				context: context
			});
	
	try {
		let res = await axios.post('http://127.0.0.1:5000/predict', payload, 
		{ headers: {"Content-Type": "application/json; charset=utf-8"}});
		
		let result = res.data;
		return result
	}
	catch(e) {
		console.log(e)
		throw createError(520, strings.err_chat_fail);
	}
}

module.exports.predict = predict;
