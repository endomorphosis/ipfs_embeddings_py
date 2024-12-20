from setuptools import setup, find_packages
setup(
    name="ipfs_embeddings_py",
	version='0.0.24',
	packages=[
		'ipfs_embeddings_py',
        'ipfs_embeddings_py.huggingface',
        'ipfs_embeddings_py.node_parser',
	],
	install_requires=[
        'transformers',
        'numpy',
        'urllib3',
        'requests',
        'boto3',
	]
)