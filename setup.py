from setuptools import setup

setup(
    name="ipfs_embeddings_py",
	version='0.0.8',
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