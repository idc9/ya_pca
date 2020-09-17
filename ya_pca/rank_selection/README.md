# PCA rank selection methods

1. Random matrix theory based singular value thresholding  (Gavish and Donoho, 2014)
	
	a. Gavish Donoho threshold
	
	b. Marcenko Pasture edge


2. Cross-validation (Perry, 2009; Owen and Perry, 2009)
	
	a. Bi-Cross-validation
	
	b. Wold hold outs (uses a [SVD imputation algorithm](https://gist.github.com/ahwillia/65d8f87fcd4bded3676d67b55c1a3954))
	
3. BIC (Bai and Ng, 2002)

4. Horn’s Parallel Analysis  (Horn, 1965; Dobriban, 2017)

5. Profile Likelihood (Zhu and Ghodsi, 2005)

6. (Minka, 2001)

8. Cutoff based on variance explained


### Noise Estimates

Some of the PCA rank selection methods require a noise estimate. We make the following available.

1. Marcenko Pastur median based estimate (Gavish and Donoho, 2014)

2. Soft Impute cross-validation (Choi et al., 2017)
	- Makes uses of SoftImpute algorithm (Mazumder et al. 2010) algorithm kindly implemented in [fancyimpute](https://github.com/iskandr/fancyimpute)




# References

Horn, J. L. (1965). [A rationale and test for the number of factors in factor analysis](https://link.springer.com/article/10.1007%252FBF02289447). Psychometrika, 30(2), 179-185.


Minka, T. P. (2001). [Automatic choice of dimensionality for PCA](https://papers.nips.cc/paper/1853-automatic-choice-of-dimensionality-for-pca.pdf). In Advances in neural information processing systems (pp. 598-604).


Bai, J., & Ng, S. (2002). [Determining the number of factors in approximate factor models](https://onlinelibrary.wiley.com/doi/pdf/10.1111/1468-0262.00273). Econometrica, 70(1), 191-221.


Zhu, M., & Ghodsi, A. (2006). [Automatic dimensionality selection from the scree plot via the use of profile likelihood](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.90.3768&rep=rep1&type=pdf). Computational Statistics & Data Analysis, 51(2), 918-930.


Perry, P. O. (2009). [Cross-validation for unsupervised learning](https://arxiv.org/pdf/0909.3052.pdf). PhD Thesis.


Owen, A. B., & Perry, P. O. (2009).[Bi-cross-validation of the SVD and the nonnegative matrix factorization](https://projecteuclid.org/euclid.aoas/1245676186).  The annals of applied statistics, 3(2), 564-594.


Mazumder, R., Hastie, T., & Tibshirani, R. (2010). [Spectral Regularization Algorithms for Learning Large Incomplete
Matrices](https://web.stanford.edu/~hastie/Papers/mazumder10a.pdf). The Journal of Machine Learning Research, 11, 2287-2322.


Gavish, M., & Donoho, D. L. (2014). [The Optimal Hard Threshold for Singular Values is 4 over root 3](https://arxiv.org/abs/1305.5870). IEEE Transactions on Information Theory, 60(8), 5040-5053.

Choi, Y., Taylor, J., & Tibshirani, R. (2017). [Selecting the number of principal components: Estimation of the true rank of a noisy matrix](https://projecteuclid.org/download/pdfview_1/euclid.aos/1513328584). The Annals of Statistics, 45(6), 2590-2617.

Dobriban, E. (2017). [Permutation methods for factor analysis and PCA](https://arxiv.org/pdf/1710.00479.pdf).


Williams, A. (2018). [How to cross-validate PCA, clustering, and matrix decomposition models](http://alexhwilliams.info/itsneuronalblog/2018/02/26/crossval/)