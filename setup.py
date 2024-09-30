from setuptools import setup

setup(
    name="ipfs_embeddings_py",
	version='0.0.13',
	packages=[
		'ipfs_embeddings_py',
	],
	install_requires=[
        'transformers',
        'numpy',
        'urllib3',
        'requests',
        'boto3',
	]
)