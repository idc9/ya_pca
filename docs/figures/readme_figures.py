from ya_pca.PCA import PCA
from ya_pca.toy_data import rand_factor_model

from ya_pca.viz import scree_plot
import matplotlib.pyplot as plt

# sample data from a factor model with 10 PCA components
X = rand_factor_model(n_samples=200, n_features=100,
                      rank=10, m=2, random_state=1)[0]

pca = PCA(n_components='rmt_threshold',
          rank_sel_kws={'thresh_method': 'mpe'})
pca.fit(X)

print('Marcenko Pastur singular value threshold selected rank:',
      pca.n_components_)


plt.figure(figsize=(8, 8))
true_rank = 10
thresh = pca.rank_sel_out_['thresh']
scree_plot(pca.all_svals_, color='black')
plt.axhline(thresh, color='red', label='MP singular value threshold')
plt.axvline(pca.n_components_,
            label='Estimated rank {}'.format(pca.n_components_), color='red')
plt.axvline(true_rank, label='True rank {}'.format(true_rank),
            color='blue', ls='--')
plt.legend()
plt.xlabel("PCA component")
plt.ylabel("Singular value")
plt.ylim(0)
plt.savefig("scree_plot.png", dpi=200, bbox_inches='tight')
