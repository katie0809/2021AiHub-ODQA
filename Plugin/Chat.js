const createError = require('http-errors');
const strings = require('../config/strings');
const logger = require('./Logger');
const elasticsearch = require('./Elasticsearch');
const pytensor = require('./Pytensor');

class Chat {
	constructor(successCallback, errCallback) {
		if (successCallback === null || errCallback === undefined) {
			throw createError(405, strings.err_wrong_params);
		}
		this.successCallback = successCallback;
		this.errCallback = errCallback;

		/** 단말 -> 서버 전송필수값(키: 길이) */
		this.requiredClientParams = { userRequest: 1024 };
	}

	/**
	 * 요청 파라미터에 대한 타입과 길이 유효성 체크한다
	 * @param {Object} param 요청 파라미터
	 */
	async checkParamValid(param) {
		Object.entries(this.requiredClientParams).map(([key, value]) => {
			if (!Object.prototype.hasOwnProperty.call(param, key))
				throw createError(405, strings.err_no_params(key));
			else if (param[key].length > value)
				throw createError(405, strings.err_wrong_params(key));

			return true;
		});

		// 모든 값 유효하면 최초 질의 저장한다
		this.originalmsg = param['userRequest'].utterance;

		// Elasticsearch 조회 후 결과를 context로 회신
		// this.context = await elasticsearch.search(this.originalmsg);

		this.context_list = await elasticsearch.search(this.originalmsg);

		this.contexts = [];
		this.questions = [];

		this.context_list.forEach(ctx => {
			
			let context = ctx["context"]
			// 줄바꿈제거
			context = context.replace(/\s+/g, ' ').trim();
			// 특수문자 제거
			var reg = /[\{\}\[\]\/?.,;:|\)*~`!^\-_+<>@\#$%&\\\=\(\'\"]/gi
			context = context.replace(reg, "");  

			let stride = 200
			let context_size = context.length
			let split_size = 3072
			let split_cnt = Math.ceil(context_size / split_size)
			logger.debug("context_size ",context_size)
			logger.debug("split_cnt ",split_cnt)

			for (var step = 0; step < split_cnt; step++) {
				let start = step * split_size
				let end = (step + 1) * split_size
				start = (start - stride >= 0 ? start - stride : start)
				end = (end + stride <= context_size  ? end + stride : context_size)
				this.contexts.push(context.substring(start,end));
				this.questions.push(ctx["question"]);
				
				logger.debug("question ", ctx["question"])
			}
		});
		// Context 를 여러개로 쪼개서 질문수행
		// logger.debug(typeof(this.context))
		// let stride = 500
		// let context_size = this.context.length
		// let split_size = 2046
		// let split_cnt = parseInt(context_size / split_size)
		// logger.debug("context_size ",context_size)
		// logger.debug("split_cnt ",split_cnt)

		// for (var step = 0; step < split_cnt; step++) {
		// 	let start = step * split_size
		// 	let end = (step + 1) * split_size
		// 	start = (start - stride >= 0 ? start - stride : start)
		// 	end = (end + stride <= context_size  ? end + stride : context_size)
			
		// 	this.answer = await pytensor.predict(this.originalmsg, this.context.substring(start,end))
		// 	if (this.answer != "") {
		// 		break;
		// 	}
		// }

		this.answer = await pytensor.predict(this.contexts, this.questions)
		
		logger.debug("answer ", this.answer)
		//TODO 전달온 Answer 중 최고 높은 점수만 회신하기
		
		// Question, Context 를 tensorflow 모델로 수행
		// this.answer = await pytensor.predict(this.originalmsg, this.context_list)
		
		// this.successCallback(this.originalmsg, '성공');
		// this.successCallback(this.originalmsg, this.answer);
		this.successCallback(this.originalmsg, this.answer.toString());
	}
}

module.exports = Chat;
