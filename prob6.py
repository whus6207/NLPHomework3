import matplotlib.pyplot as plt

from dependencyRNN import *
from sklearn.manifold import TSNE

ans_emb = dependencyRNN.load('random_init.npz')
X = ans_emb.answers
X = [list for key, list in X.iteritems()]
tsne = TSNE(n_components=2,perplexity=30.0)
X_reduced = tsne.fit_transform(X)
plt.scatter(X_reduced[:,0],X_reduced[:,1])
plt.show()
