import setuptools

setuptools.setup(
    name = 'AutomaticDifferentiation',
    version = '0.0.1',
    author = 'Haoming Chen, Yiting Han, Ivonne Martinez, Gahyun Sung',
    author_email = '',
    description = 'Function Automatic Automatic Differentiation'
    long_description=open('docs/milestone1/README.md').read()
    long_description_content_type="text/markdown",
    python_requires='>=3.6',
    url = 'https://github.com/cs107-sandbox/cs107-FinalProject',
    install_requires=['numpy', 'pytest', 'pip'],
    license ='MIT',
    packages=['src'],
    install_requires = ['numpy']

)
