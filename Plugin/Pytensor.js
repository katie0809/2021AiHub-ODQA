const strings = require('../config/strings');
const logger = require('./Logger');

const axios = require('axios');
const FormData = require('form-data');

async function predict(question, context) {

	let payload = JSON.stringify({
		question: question,
		context: context
	});

    let res = await axios.post('http://127.0.0.1:5000/predict', payload, 
        { headers: {"Content-Type": "application/json; charset=utf-8"}});

    let data = res.data;
    console.log(data);

	return data
}

module.exports.predict = predict;
