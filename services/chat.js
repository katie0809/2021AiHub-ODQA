const express = require('express');
const createError = require('http-errors');

const strings = require('../config/strings');

const logger = require('../Plugin/Logger');
const ChatHandler = require('../Plugin/Chat');

const router = express.Router();
const puppeteer = require('puppeteer');
const { unlinkSync } = require('fs');
/** 카드 html */
const card_html=`
<html>
	<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
	<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate" />
	<meta http-equiv="Pragma" content="no-cache" />
	<meta http-equiv="Expires" content="0" />
	<head>
		<title>Insert title here</title>
		<style>
		@charset "utf-8";
		/*@import url('font-awesome.min.css');*/
		html {
			height: 100%;
		}
		body {
			position: relative;
			max-width: 340px;
			min-height: 100%;
			font-family: "Nanum Gothic", Helvetica Neue, Helvetica, Arial, sans-serif;
			color: #444;
			font-size: 14px;
			line-height: 1.5;
			letter-spacing: -0.6px;
			background-color: #f2f4f7;
			transition: all .3s;
		}
		.container {
			width: 1170px;
			max-width: none !important;
			transition: all .3s;
		}
		body.width-fluid,
		body.width-fluid .container {
			width: 100% !important;
		}

		*,
		*::after,
		*::before {
			box-sizing: border-box;
		}

		::selection {
			background: #2AFEF1;
		}

		.footer {
			color: #888;
			font-size: 12px;
			padding: 32px 0;
			margin-top: 40px;
			border-top: 1px solid #e3e7ee;
			background-color: #fff;
		}
		/*.footer ul > li .fa {
			position: relative; top: 1px;
			margin-right: 4px;
		}*/

		/* a */
		a {
			cursor: pointer;
			color: #0088f5;
		}
		a:focus {
			outline: 0;
		}
		a:hover {
			outline: 0;
		}

		/* btn */
		.btn.active.focus, .btn.active:focus,
		.btn.focus, .btn:active.focus,
		.btn:active:focus, .btn:focus {
			outline: none;
		}
		.btn.active, .btn:active {
			box-shadow: none;
			background-color: transparent;
		}
		.btn-area {
			font-size: 0;
		}
		.btn-area .btn + .btn {
			margin-left: 8px;
		}
		.btn-default.active, .btn-default:active
		.btn-default.focus, .btn-default:focus,
		.btn-default:hover {
			background-color: inherit;
		}
		.btn-more,
		.btn-more.active, .btn-more:active
		.btn-more.focus, .btn-more:focus,
		.btn-more:hover {
			border-color: transparent;
			background-color: transparent;
		}
		.btn-more.active .caret {
			border-top: 0;
			border-bottom: 4px solid;
		}

		/* fontawesome */
		i,
		.fa {
			font-size: 14px;
		}
		.fa.fa-2 {
			font-size: 2em !important;
		}
		/* card */
		.card {
			margin-bottom: 8px;
			border: 1px solid #ddd;
			border-radius: 4px;
			background-color: #fff;
		}
		.card:last-child {
			margin-bottom: 0;
		}
		.card .card-header {
			display: inline-block;
			width: 100%;
			padding: 8px 16px;
			border-bottom: 1px solid #eee;
			border-top-left-radius: 4px;
			border-top-right-radius: 4px;
			background-color: #f9f9f9;
		}
		.card .card-header .card-title {
			font-weight: 600;
			display: inline;
		}
		.card .card-header .badge {
			position: relative; top: 2px;
			padding: 2px 7px 3px;
			float: right;
			background-color: #28a745;
		}
		.card .card-body {
			display: inline-block;
			width: 100%;
			padding: 8px 16px;
		}
		.card .card-body p {
			color: #888;
			font-size: 14px;
		}
		.card .card-body .btn-more {
			float: right;
		}

		/* card-icon */
		.card.card-icon .card-body {
			position: relative;
			padding: 20px 25px 20px 70px;
		}
		.card.card-icon .card-body .card-title {
			font-weight: 600;
		}
		.card.card-icon .card-body .fa {
			position: absolute; top: 20px; left: 24px;
		}
		.card.card-icon .card-body .card-title {
			font-size: 16px;
			margin-bottom: 8px;
		}

		/* cont-group */
		.cont-group + .cont-group {
			margin-top: 32px;
		}
		.cont-title {
			position: relative;
			font-size: 18px;
			font-weight: 700;
			padding-left: 12px;
			margin-bottom: 12px;
		}
		.cont-title:after {
			content: '';
			position: absolute; top: 50%; left: 0;
			display: inline-block;
			width: 4px;
			height: 16px;
			margin-top: -8px;
			background-color: #007bff;
			border-radius: 2px;
		}

		/* list-step */
		.list-step li {
			position: relative;
			text-align: center;
			padding-top: 60px;
			margin: 8px 0 24px;
		}
		.list-step li .num {
			position: absolute; top: 0; left: 15px; right: 15px;
			font-family: arial;
			color: #c4e1ff;
			font-size: 40px;
			font-weight: 800;
			text-align: center;
		}
		.list-step li strong {
			font-weight: 800;
			text-decoration: underline;
		}
		.list-step li:after {
			content: '';
			position: absolute; top: 50%; right: 0;
			display: inline-block;
			width: 24px;
			height: 45px;
			margin-top: -22px;
			margin-right: -13px;
			background: url(../images/common/step-arrow.png) 0 0 no-repeat;
		}
		.list-step li:last-child:after {
			content: none;
		}


		#btn-viewport .viewport-fixed,
		#btn-viewport.active .viewport-fluid,
		#btn-recommend .recommend-show,
		#btn-recommend.active .recommend-hide {
			display: none;
		}
		#btn-viewport .viewport-fluid,
		#btn-viewport.active .viewport-fixed,
		#btn-recommend .recommend-hide,
		#btn-recommend.active .recommend-show {
			display: block;
		}
		#btn-viewport .fa,
		#btn-recommend .fa {
			position: relative; top: 1px;
			color: #ccc;
			font-weight: 14px;
			margin-right: 6px;
		}
		#btn-viewport.active .fa,
		#btn-recommend.active .fa {
			color: #007bff;
		}
		@media (max-width: 1169px) {
			#btn-viewport {
				display: none
			}
		}
		</style>
	</head>
	<body>
		<div class="card"><div class="card-header">
`

/** 질의 답변 요청 */
// 1. 답변과 논문 정보를 반환한다.
router.post('/', (req, res, next) => {
	try {
		// 요청객체 유효성 체크
		if (!req || !res || !req.body) {
			next(createError(405, strings.err_wrong_params));
			return;
		}
		let return_type = req.body['contexts'][0]['params']['type_name']['value'];
		let successCallback;
		if(return_type == '논문정보형') {
			successCallback = (request, topanswer, contexts, answers) => {
				const url = 'https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn='
				let carditems = []
				contexts.forEach((context, i)=> {
					curanswer = answers[i]['answer'][0]['answer']
					if(curanswer.length > 0) {
						cardinfo = {
							"imageTitle": {
								"title": curanswer
							},
							"itemList": [{
								"title": "참고논문",
								"description": `${context['title']}(${context['year']})`
							},
							{
								"title": "저자",
								"description": context['authors']
							}],
							"itemListAlignment": "left",
							"buttons": [
								{
									"label": "논문보러가기",
									"action": "webLink",
									"webLinkUrl": `${url}${context['doc_id']}`
								}
							]
						}
						carditems.push(cardinfo)
					}
				});
				const resBody = {
					"version": "2.0",
					"template": {
					  "outputs": [
						{
							simpleText: {
								"text": "제가 찾은 답변들이랍니다.."
							},
						},
						{
						  "carousel": {
							"type": "itemCard",
							"items": carditems,
						  }
						}],
						"quickReplies": [
							{
								"messageText": "도움돼요",
								"action": "message",
								"label": "도움돼요"
							},
							{
								"messageText": "별로에요",
								"action": "message",
								"label": "별로에요",
								"data": {
									"extra": {
										"answer": topanswer['answer']
									}
								}
							}
						]
					}
				  }
				res.status(200).json(resBody);
				logger.debug('RESPONSE', resBody, req.originalUrl);
			}
		}
		else if(return_type == '이미지형') {
			successCallback = async (i, topanswer, contexts, answers) => {
				let curctx = contexts[i];
				let answer_st = curctx['context'].indexOf(topanswer['answer'])
				console.log(answer_st)
				if(answer_st < 0) answer_st = 0
				let answer_end = answer_st+topanswer['answer'].length
				let contentHtml = `${card_html}				
					<h4 class="card-title">${curctx['title']}</h4></div>
					<div class="card-body"><p>${curctx['context'].substr(answer_st-120, 120)}<mark>${topanswer['answer']}</mark>${curctx['context'].substr(answer_end, 200)}</p></div></div></body></html>`
				const browser = await puppeteer.launch({args: ['--start-fullscreen', '--no-sandbox', '--disable-setuid-sandbox']}); // --start-fullscreen 옵션 추가
				const page = await browser.newPage();
				await page.setViewport({width: 340, height: 380}); // 변경
				await page.setContent(contentHtml);
				await page.screenshot({path: 'img.png', omitBackground: true});
				await browser.close();

				const resBody = {
					version: '2.0',
					template: {
						outputs: [
							{
								"simpleImage": {
									"imageUrl": "http://34.217.138.255:8080/chat/imagedown",
									"altText": "보물상자입니다"
								}
							},
						],
					},
				};

				res.status(200).json(resBody);
				logger.debug('RESPONSE', resBody, req.originalUrl);
			}

		}
		else {
			successCallback = (request, topanswer, contexts, answers) => {
				const url = 'https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn='
				let carditems = []
				contexts.forEach((context)=> {
					cardinfo = {
						"title": context['title'],
						"description": `${context['authors']}(${context['year']})`,
						"link": {
							"web": `${url}${context['doc_id']}`
						}
					}
					carditems.push(cardinfo)
				});
				const resBody = {
					"version": "2.0",
					"template": {
					  "outputs": [
						{
							simpleText: {
								"text": "제가 생각하기에 질문의 답변은..\n\n"+topanswer['answer']
								// "text": "제가 생각하기에 질문의 답변은..\n\n"+answers[0]['answer'][0]['answer']
							},
						},
						{
						  "listCard": {
							"header": {
							  "title": "관련 논문 리스트"
							},
							"items": carditems,
						  }
						}],
						"quickReplies": [
							{
								"messageText": "도움돼요",
								"action": "message",
								"label": "도움돼요"
							},
							{
								"messageText": "별로에요",
								"action": "message",
								"label": "별로에요",
								"data": {
									"extra": {
										"answer": topanswer['answer']
									}
								}
							}
						]
					}
				  }
				res.status(200).json(resBody);
				logger.debug('RESPONSE', resBody, req.originalUrl);
			}

		}
		const chat = new ChatHandler(
			successCallback,
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

// 2. 스코어 높은 답변 이미지를 반환한다
router.post('/answers', (req, res, next) => {
	try {
		// 요청객체 유효성 체크
		if (!req || !res || !req.body) {
			next(createError(405, strings.err_wrong_params));
			return;
		}

		// 2. 스코어 높은 답변 이미지를 반환한다
		const chat = new ChatHandler(
			async (request, topanswer, contexts, answers) => {
			// async (request, answer, context) => {
				console.log(Object.keys(contexts))
				let answer_st = contexts['context'].indexOf(topanswer['answer'])
				if(answer_st < 0) answer_st = 0
				let answer_end = answer_st+topanswer['answer'].length
				let contentHtml = `${card_html}				
					<h4 class="card-title">${contexts['title']}</h4></div>
					<div class="card-body"><p>${contexts['context'].substr(answer_st-100, 100)}<mark>${topanswer['answer']}</mark>${contexts['context'].substr(answer_end, 200)}</p></div></div></body></html>`
				const browser = await puppeteer.launch({args: ['--start-fullscreen', '--no-sandbox', '--disable-setuid-sandbox']}); // --start-fullscreen 옵션 추가
				const page = await browser.newPage();
				await page.setViewport({width: 340, height: 380}); // 변경
				await page.setContent(contentHtml);
				await page.screenshot({path: 'img.png', omitBackground: true});
				await browser.close();

				const resBody = {
					version: '2.0',
					template: {
						outputs: [
							{
								"simpleImage": {
									"imageUrl": "http://34.217.138.255:8080/chat/imagedown",
									"altText": "보물상자입니다"
								}
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
	} catch (e) {
		next(e);
	}

});

// 3. 스코어 높은 답변 이미지를 반환한다
router.post('/image', (req, res, next) => {
	try {
		// 요청객체 유효성 체크
		if (!req || !res || !req.body) {
			next(createError(405, strings.err_wrong_params));
			return;
		}

		// 2. 스코어 높은 답변 이미지를 반환한다
		const chat = new ChatHandler(
			async (request, topanswer, contexts, answers) => {
			// async (request, answer, context) => {
				console.log(Object.keys(contexts))
				let answer_st = contexts['context'].indexOf(topanswer['answer'])
				if(answer_st < 0) answer_st = 0
				let answer_end = answer_st+topanswer['answer'].length
				let contentHtml = `${card_html}				
					<h4 class="card-title">${contexts['title']}</h4></div>
					<div class="card-body"><p>${contexts['context'].substr(answer_st-100, 100)}<mark>${topanswer['answer']}</mark>${contexts['context'].substr(answer_end, 200)}</p></div></div></body></html>`
				const browser = await puppeteer.launch({args: ['--start-fullscreen', '--no-sandbox', '--disable-setuid-sandbox']}); // --start-fullscreen 옵션 추가
				const page = await browser.newPage();
				await page.setViewport({width: 340, height: 380}); // 변경
				await page.setContent(contentHtml);
				await page.screenshot({path: 'img.png', omitBackground: true});
				await browser.close();

				const resBody = {
					version: '2.0',
					template: {
						outputs: [
							{
								"simpleImage": {
									"imageUrl": "http://34.217.138.255:8080/chat/imagedown",
									"altText": "보물상자입니다"
								}
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
	} catch (e) {
		next(e);
	}

});

/** 이미지 요청 */
router.get('/imagedown', (req, res, next) => {
	try {
		res.download('img.png', 'img.png', (err) => {
			// 전송후 이미지 삭제
			if (!err) {
				unlinkSync('img.png');
				res.end();
			}
		})
	} catch (e) {
		console.log(e)
		next(e);
	}
});

/** 응답 결과 점수 */
router.post('/score', (req, res, next) => {
	try {
		logger.debug('REQUEST', req.body, req.originalUrl);

		const resBody = {
			version: '2.0',
			template: {
				outputs: [
					{
						"simpleText": {
							"text": "보내주신 응답은 서비스를 향상시키는데에 사용됩니다, 감사합니다",
						}
					},
				],
			},
		};

		res.status(200).json(resBody);
		logger.debug('RESPONSE', resBody, req.originalUrl);
		
	} catch (e) {
		console.log(e)
		next(e);
	}
});

/** 질의 토큰화 결과 요청 */
// router.post('/casualtalk', (req, res, next) => {
// 	console.log('in casual')
// 	try {
// 		logger.debug('REQUEST', req.body, req.originalUrl);

// 		// Casual Talk
// 		const chat = new ChatHandler(
// 			(answer) => {
// 				console.log('get talk res')
// 				const resBody = {
// 					"version": "2.0",
// 					"template": {
// 						"outputs": [
// 						{
// 							simpleText: {
// 								"text": answer
// 							}
// 						}]
// 					}
// 				}
// 				res.status(200).json(resBody);
// 				logger.debug('RESPONSE', resBody, req.originalUrl);
// 			},
// 			(error) => {
// 				next(createError(520, strings.err_chat_fail));
// 			}
// 		);
// 		chat.getCasualTalkResponse(req.body);
		
// 	} catch (e) {
// 		console.log(e)
// 		next(e);
// 	}
// });
module.exports = router;
