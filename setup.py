import setuptools

with open('README.md') as f:
    readme = f.read()

setuptools.setup(
    name='allnews_am',
    version='0.1',
    description='Natural Language Processing for Armenian News Summarization.',
    long_description=readme,
    packages=setuptools.find_packages(),
    include_package_data=True,
)
