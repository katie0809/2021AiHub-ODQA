const strings = require('../config/strings');
const logger = require('./Logger');

const { Client } = require('@elastic/elasticsearch');
const client = new Client({ node: 'http://localhost:9200' });

async function search(question) {
	// 질문 예시 : '사회투자의 관점에서는 잊고 있는 다른 더 큰 특징은 무엇인가?'
	logger.debug('question', question);

	const { body } = await client.search({
		index: 'qa-index',
		// from: 20,
		// size: 10,
		body: {
			query: {
				match: {
					context: question,
				},
			},
		},
	});

	if (!body || !body.hits || !body.hits.hits || body.hits.hits.length == 0) {
		return '조회결과가 없습니다.';
	} else {
		logger.debug(
			'Elasticsearch Search Result',
			body.hits.hits[0]._source.context
		);
		return body.hits.hits[0]._source.title;
	}
}

module.exports.search = search;
