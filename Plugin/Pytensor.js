const strings = require('../config/strings');
const logger = require('./Logger');
const createError = require('http-errors');
const axios = require('axios');
const FormData = require('form-data');

const instance = axios.create();
instance.defaults.timeout = 5000; //3sec

async function predict(context, question) {

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

async function casualtalk(question) {

	let payload = JSON.stringify({
				question: question,
			});
	
	try {
		let res = await axios.post('http://127.0.0.1:5000/casualtalk', payload, 
		{ headers: {"Content-Type": "application/json; charset=utf-8"}});
		
		let result = res.data;
		return result
	}
	catch(e) {
		console.log(e)
		throw createError(520, strings.err_chat_fail);
	}
}

async function classify(question) {

	// 2. 1개 서버에 전체 전달 후 답변 받는 경우
	let payload = JSON.stringify({
				question: question,
			});
	
	try {
		let res = await axios.post('http://127.0.0.1:5000/intent', payload, 
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
module.exports.casualtalk = casualtalk;
