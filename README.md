# Yet another PCA package

A sklearn compatible python package for principal components analysis that includes several methods for PCA rank selection such as [random matrix theory based thresholds](https://arxiv.org/abs/1305.5870), [Wold style and bi-cross validation](https://projecteuclid.org/euclid.aoas/1245676186), [Minka's method](https://papers.nips.cc/paper/1853-automatic-choice-of-dimensionality-for-pca.pdf), [Horn's Parallel Analysis](), etc. See [here](pca/rank_selection/README.md) for the list of currently supported rank selection methods as well as the corresponding references.


## Installation

<!--
```
pip install pca (coming soon!)
```
-->

```
git clone https://github.com/idc9/pca.git
python setup.py install
```

## Example

```python
from pca.PCA import PCA
from pca.toy_data import rand_factor_model

# sample data from a factor model with 10 PCA components
X = rand_factor_model(n_samples=200, n_features=100,
                      rank=10, m=2, random_state=1)[0]

# fit PCA and select the rank by thresholding
# the singular values using the Marcenko Pastur distribution
pca = PCA(n_components='rmt_threshold',
          rank_sel_kws={'thresh_method': 'mpe'})
pca.fit(X)
```
<!--

```
print('Marcenko Pastur singular value threshold selected rank:', pca.n_components_)

'Marcenko Pastur singular value threshold selected rank: 10'
```

![PCA scree plot](/docs/figures/scree_plot.png)
-->


<img src="/docs/figures/scree_plot.png" width="400" height="400">


# Help and support

Additional documentation, examples and code revisions are coming soon. For questions, issues or feature requests please reach out to Iain: <idc9@cornell.edu>.

<!--
## Testing
Testing is done using nose.
-->

## Contributing

We welcome contributions to make this a stronger package: data examples, bug fixes, spelling errors, new features, etc.

# Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4091759.svg)](https://doi.org/10.5281/zenodo.4091759)



