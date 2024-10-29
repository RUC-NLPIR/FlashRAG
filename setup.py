from setuptools import setup, find_packages

with open("requirements.txt",encoding='utf-8') as fp:
    requirements = fp.read().splitlines()
with open("README.md", "r",encoding='utf-8') as fh:
    long_description = fh.read()

extras_require = {
    'core': requirements,
    'retriever': ['pyserini', 'sentence-transformers>=3.0.1'],
    'generator': ['vllm>=0.4.1']
}
extras_require['full'] = sum(extras_require.values(), [])

setup(
    name="flashrag-dev",
    version="0.1.2",
    packages=find_packages(),
    url="https://github.com/RUC-NLPIR/FlashRAG",
    license="MIT License",
    author="Jiajie Jin, Yutao Zhu, Chenghao Zhang, Xinyu Yang, Zhicheng Dou",
    author_email="jinjiajie@ruc.edu.cn",
    description="A library for efficient Retrieval-Augmented Generation research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    package_data={"flashrag-dev": ["/config/basic_config.yaml"]},
    install_requires=extras_require['core'],
    extras_require=extras_require,
    python_requires=">=3.9",
)
