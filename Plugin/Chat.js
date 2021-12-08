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
		this.requiredClientParams = { userRequest: 512 };

		/** 모델 설정값 */
		this.modelConfig = {
			'max_seq_len': 4096,
			'doc_stride': 64
		}
	}

	async getCasualTalkResponse(param) {
		try {
			this.answer = await pytensor.casualtalk(param['userRequest'].utterance)
			this.successCallback(this.answer);
		}
		catch(e) {
			this.errCallback(e);
		}
	}

	/**
	 * @param {Array} predicted 
	 */
	getTopAnswer(predicted) {
		let score = 0;
		let topi = 0;
		let topj = 0;
		let topanswer;
		predicted.forEach((answers, i) => {
			answers['answer'].forEach((pred, j) => {
				// console.log(pred, i, j)
				let curscore = (pred['answer_score'][0] + pred['answer_score'][1]) / 2;
				if(curscore > score) {
					score = curscore
					topanswer = pred;
					topi = i;
					topj = j;
				}
			});
		})
		// console.log("RETURN TOP ANSWER", topanswer, topi, topj)
		return [topanswer, topi, topj];
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

		try {
			// 질의와 관련있는 논문 검색
			this.context_list = await elasticsearch.search(this.originalmsg);
			this.contexts = [];
			this.questions = [];
			
			if(this.context_list.includes('조회결과가 없습니다')) {
				this.errCallback();
			}
			else{
				this.context_list.forEach(ctx => {
				
					// 논문 텍스트 전처리 - 특수문자 제거
					let context = ctx["context"]
					context = context.replace(/\s+/g, ' ').trim();
					var reg = /[\{\}\[\]\/?.,;:|\)*~`!^\-_+<>@\#$%&\\\=\(\'\"]/gi
					context = context.replace(reg, "");  
		
					let stride = 200
					let context_size = context.length
					let split_size = 3072
					let split_cnt = Math.ceil(context_size / split_size)
					this.contexts.push(context.substring(500, 4500));
					this.questions.push(ctx["question"]);

				});
				
				// 논문에서 예상 정답 추출
				this.answer = await pytensor.predict(this.contexts, this.questions)

				// 1.스코어가 가장 높은 정답텍스트 하나만 반환한다.
				// this.successCallback(this.originalmsg, this.getTopAnswer(this.answer));

				// 2.스코어가 가장 높은 정답블록과 문서블록 반환한다.
				// let [topanswer, i, j] = this.getTopAnswer(this.answer);
				// this.successCallback(this.originalmsg, topanswer, this.context_list[i]);

				// 3.스코어가 가장 높은 정답블록과 전체 문서블록 반환한다.
				let [topanswer, i, j] = this.getTopAnswer(this.answer);
				this.successCallback(i, topanswer, this.context_list, this.answer);

			}
		}
		catch(e) {
			this.errCallback(e);
		}
	}
}

module.exports = Chat;
