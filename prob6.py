import matplotlib.pyplot as plt

from dependencyRNN import *
from sklearn.manifold import TSNE

ans_emb = DependencyRNN.load('random_init.npz')
ans = ans_emb.answers
X = []
word = []
count = 0
for key, x in ans.iteritems():
	X.append(x)
	if count % 20 == 0:
		word.append(key)
	else:
		word.append('')
	count += 1

tsne = TSNE(n_components=2, perplexity=30.0)
X_reduced = tsne.fit_transform(X)

fig, ax = plt.subplots()
ax.scatter(X_reduced[:,0],X_reduced[:,1])

for i, txt in enumerate(word):
    ax.annotate(txt, (X_reduced[i,0],X_reduced[i,1]))

plt.show()
