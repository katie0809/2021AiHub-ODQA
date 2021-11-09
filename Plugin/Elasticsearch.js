const strings = require('../config/strings');
const logger = require('./Logger');

const { Client } = require('@elastic/elasticsearch');
const client = new Client({ node: 'http://localhost:9200' });

async function search(question) {
	logger.debug('elasticsearch question', question);

	const { body } = await client.search({
		index: 'qa-index-nori-anly',
		body: {
			query: {
				match: {
					'context.nori': question,
				},
			},
		},
	});

	if (!body || !body.hits || !body.hits.hits || body.hits.hits.length == 0) {
		return '조회결과가 없습니다.';
	} else {
		logger.debug(
				'Elasticsearch Search Result Title : ',
				body.hits.hits[0]._source.title
			);
		return body.hits.hits[0]._source.context;
	}
}

module.exports.search = search;
