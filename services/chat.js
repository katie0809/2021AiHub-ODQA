const express = require('express');
const createError = require('http-errors');

const strings = require('../config/strings');

const logger = require('../Plugin/Logger');
const ChatHandler = require('../Plugin/Chat');

const router = express.Router();

/** 질의 답변 요청 */
router.post('/', (req, res, next) => {
	try {
		// 요청객체 유효성 체크
		if (!req || !res || !req.body) {
			next(createError(405, strings.err_wrong_params));
			return;
		}

		const chat = new ChatHandler(
			(request, predicted) => {
				const resBody = {
					version: '2.0',
					template: {
						outputs: [
							{
								simpleText: {
									"text": predicted
								},
							},
						],
					},
				};

				res.status(200).json(resBody);
				logger.debug('RESPONSE', resBody, req.originalUrl);
			},
			(error) => {
				next(createError(520, strings.err_chat_fail));
			}
		);

		logger.debug('REQUEST', req.body, req.originalUrl);
		chat.checkParamValid(req.body);
	} catch (e) {
		next(e);
	}
});

/** 질의 토큰화 결과 요청 */
router.post('/tokens', (req, res, next) => {
	try {
	} catch (e) {
		next(e);
	}
});
module.exports = router;
