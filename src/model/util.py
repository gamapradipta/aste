import constant as c

class Decoder(object):
	def get_spans(self, tags, token_ranges, tag_type):
		spans = []
		begin = -1
		length = len(token_ranges)
		for i, (l, r) in enumerate(token_ranges):
			if tags[l][l] == c.IGNORE_INDEX:
				continue
			elif tags[l][l] == tag_type:
				if begin == -1:
					begin = i
			elif tags[l][l] != tag_type:
				if begin != -1:
					spans.append([begin, i - 1])
					begin = -1
		if begin != -1:
			spans.append([begin, length - 1])
		return spans

	def find_triplets(self, tags, token_ranges, aspect_spans, sentiment_spans):
		triplets = []
		# Check Word Relation Between Extracted Aspect and Sentiment Spans Only
		for al, ar in aspect_spans:
			for sl, sr in sentiment_spans:
				tag_num = [0] * 6
				for i in range(al, ar+1):
					for j in range(sl, sr+1):
						a_begin = token_ranges[i][0]
						s_begin = token_ranges[j][0]
						if al < sl:
							tag_num[int(tags[a_begin][s_begin])] += 1
						else:
							tag_num[int(tags[s_begin][a_begin])] += 1
				if sum(tag_num[3:]) == 0:
					continue
				polarity = 3 + tag_num[3:].index(max(tag_num[3:]))
				triplets.append([[al, ar], [sl, sr], polarity])
		return triplets	

	def format_spans_as_string(self, spans):
		return [self.format_span_as_string(span) for span in spans]

	def format_span_as_string(self, span):
		return '-'.join(map(str, span))

	def parse_out(self, tags, token_ranges, format_span_as_string=False):
		aspect_spans = self.get_spans(
			tags,
			token_ranges,
			c.LABELS["aspect"]
		)
		sentiment_spans = self.get_spans(
			tags,
			token_ranges,
			c.LABELS["sentiment"]
		)

		triples = self.find_triplets(
			tags, 
			token_ranges,
			aspect_spans,
			sentiment_spans
		)
		if format_span_as_string:
			return (
				self.format_spans_as_string(triples),
				self.format_spans_as_string(aspect_spans),
				self.format_spans_as_string(sentiment_spans)
			)
		return triples, aspect_spans, sentiment_spans
	
	def get_token_from_span(self, tokens, span):
		assert type(span) == list
		return tokens[span[0]:span[1]+1]

	def get_token_from_spans(self, tokens, spans):
		return [self.get_token_from_span(tokens, span) for span in spans]

	def generate_triple(self, tokens, triple):
		assert len(triple) == 3
		aspect_span, sentiment_span, polarity_label = triple
		aspect_term = self.get_token_from_span(tokens, aspect_span)
		sentiment_term = self.get_token_from_span(tokens, sentiment_span)
		polarity = list(c.LABELS.keys())[list(c.LABELS.values()).index(polarity_label)]
		return aspect_term, sentiment_term, polarity

	def generate_triples(self, tokens, triples):
		return [
			self.generate_triple(tokens, triple) for triple in triples
		]
	
	def generate_triples_from_tags(self, tokens, tags, token_ranges):
		triples, aspect_spans, sentiment_spans = self.parse_out(tags, token_ranges, format_span_as_string=False)
		return (
			self.generate_triples(tokens, triples),
			self.get_token_from_spans(tokens, aspect_spans),
			self.get_token_from_spans(tokens, sentiment_spans),
		)