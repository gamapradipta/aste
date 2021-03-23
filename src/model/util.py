import constant as c

class Decoder(object):
    def get_spans(self, tags, token_ranges, tag_type):
        spans = []
        begin = -1

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
            spans.append([begin, i - 1])
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
				triplets.append([al, ar, pl, pr, polarity])
		return triplets	

	def parse_out(self, tags, token_ranges):
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
			token_rage,
			aspect_spans,
			sentiment_spans
		)
		return triples, aspect_spans, sentiment_spans
		
		