import gensim
import json

with open('hist_split.json') as input_f:
	data = json.load(input_f)['train']
	sentences = [ [w[0] for w in sent[0]] for sent in data]

model = gensim.models.Word2Vec(size=100, window=5, min_count=1)
model.build_vocab(sentences)
alpha, min_alpha, passes = (0.025, 0.001, 20)
alpha_delta = (alpha - min_alpha) / passes
for epoch in range(passes):
	model.alpha, model.min_alpha = alpha, alpha
	model.train(sentences)
	print('completed pass %i at alpha %f' % (epoch + 1, alpha))
	alpha -= alpha_delta
	np.random.shuffle(sentences)

model.save('w2v.model')