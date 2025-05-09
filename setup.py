from setuptools import setup, find_packages

with open("requirements.txt",encoding='utf-8') as fp:
    requirements = fp.read().splitlines()
with open("README.md", "r",encoding='utf-8') as fh:
    long_description = fh.read()
version = {}
with open("flashrag/version.py", encoding="utf8") as fp:
    exec(fp.read(), version)

extras_require = {
    'core': requirements,
    'retriever': ['pyserini', 'sentence-transformers>=3.0.1'],
    'generator': ['vllm'],
    'multimodal': ['timm', 'torchvision', 'pillow', 'qwen_vl_utils']
}
extras_require['full'] = sum(extras_require.values(), [])

setup(
    name="flashrag_dev",
    version=version['__version__'],
    packages=find_packages(),
    url="https://github.com/RUC-NLPIR/FlashRAG",
    license="MIT License",
    author="Jiajie Jin, Yutao Zhu, Chenghao Zhang, Xinyu Yang, Zhicheng Dou",
    author_email="jinjiajie@ruc.edu.cn",
    description="A library for efficient Retrieval-Augmented Generation research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={"": ["*.yaml"]},
    include_package_data=True,
    install_requires=extras_require['core'],
    extras_require=extras_require,
    python_requires=">=3.9",
)
