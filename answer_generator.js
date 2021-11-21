const puppeteer = require('puppeteer');
// const fs = require('fs');
// var contentHtml = fs.readFileSync('./sample.html', 'utf8');

document=`
<html>
	<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
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
		mark {
			background-color: yellow;
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
`
`
		<div class="card"><div class="card-header"><h4 class="card-title">성종</h4><span class="badge" title="연관어">창경궁</span></div><div class="card-body"><p>세조의 큰아들인 덕종의 둘째아들로, 1469년 예종이 죽자 병약한 형 월산군을 대신하여 13살의 나이로 왕위에 올랐다. 1470년에는 국가가 농민으로부터 직접 조세를 거두어들인 다음 관리들에게 녹봉을 현물로 지급하는 관수관급제를 실시하였다. 성종은 선왕들의 통치 제도 정비 작업을 법제적으로 마무리하는 한편, 숭유억불의 정책을 더욱 굳건히 펴 나갔다. 1478년에는 홍문관에 집현전적인 기능을 편입시켜 학문 연구 기관으로 개편했다. 또한 즉위 이후 &lt;경국대전&gt;의 편찬 사업을 이어받아 1485년 이를 최종적으로 완성·반포했다. 1457(세조 3)~1494(성종 25). &lt;경국대전&gt;의 반포와 관수관급제 실시, 유학의 장려 등을 통해 조선봉건국가체제를 완성한 조선의 제9대 왕.</p></div></div>
	</body>
</html>
`

async function generate_image() {
    const browser = await puppeteer.launch({args: ['--start-fullscreen', '--no-sandbox', '--disable-setuid-sandbox']}); // --start-fullscreen 옵션 추가
    const page = await browser.newPage();
    await page.setViewport({width: 340, height: 380}); // 변경
    await page.setContent(contentHtml);
    await page.screenshot({path: 'naver.png', omitBackground: true});
    await browser.close();
}
  
rr();