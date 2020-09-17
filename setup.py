from setuptools import setup, find_packages

# def readme():
#     with open('README.rst') as f:
#            return f.read()

install_requires = ['numpy', 'pandas', 'scipy', 'scikit-learn',
                    'seaborn', 'matplotlib', 'statsmodels', 'fancyimpute']

setup(name='ya_pca',
      version='0.0.1',
      description='Yet another package for principal components analysis.'
                  'This one focuses on PCA rank selection.',
      author='Iain Carmichael',
      author_email='idc9@cornell.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=install_requires,
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
