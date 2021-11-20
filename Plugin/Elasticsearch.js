const strings = require('../config/strings');
const logger = require('./Logger');

const { Client } = require('@elastic/elasticsearch');
const client = new Client({ node: 'http://localhost:9200' });

async function search(question) {
	logger.debug('elasticsearch question', question);

	// const { body } = await client.search({
	// 	index: 'qa-index-nori-anly',
	// 	size: 3,
	// 	body: {
	// 		query: {
	// 			match: {
	// 				'context.nori': question,
	// 			},
	// 		},
	// 	},
	// });

	const { body } = await client.search({
		index: 'qa-index',
		size: 3,
		body: {
			query: {
				multi_match: {
					"query": question, 
                	"fields": [ "context" ] 
                	// "fields": [ "context.nori", "qas.keyword.keyword_text", "qas.answer.answer_text" ] 
				},
			},
		},
	});

	if (!body || !body.hits || !body.hits.hits || body.hits.hits.length == 0) {
		return '조회결과가 없습니다.';
	} else {
		// 1개 결과만 반환할 때
		// logger.debug(
		// 		'Elasticsearch Search Result Title : ',
		// 		body.hits.hits[0]._source.title
		// 	);
		// return body.hits.hits[0]._source.context;
		
		// TODO 여러개 결과를 반환할 때
		var context_list = [];
		for (var idx in body.hits.hits) {
			let doc = body.hits.hits[idx]._source
			logger.debug(
				'Elasticsearch Search Result Title : ',
				doc.title
			);

			context_list.push({
				"doc_id": doc.doc_id,
				"title": doc.title,
				"authors" : doc.authors,
				"journal" : {
					"ko" : doc.journal.ko,
					"en" : doc.journal.en
				},
				"year" : doc.year,
				"context": doc.context,
				"question":question
			});
		}
		logger.debug(
			'Elasticsearch Search Result : ',
			context_list
		);
		return context_list;
	}
}

module.exports.search = search;
