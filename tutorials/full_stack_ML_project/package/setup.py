from setuptools import setup


setup(
    name='package',
    version=1.0,
    description="Package setup for a full stack machine learning project",
    author="Irene Busah",
    author_email="i.busah123@gmail.com",
    packages=['package.feature', 'package.model_training'],
    install_requires=['numpy', 'pandas', 'scikit-learning', 'matplotlib', 'seaborn', 'mlflow']
)
