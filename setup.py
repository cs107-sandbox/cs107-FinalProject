import setuptools

setuptools.setup(
    name = 'AutomaticDifferentiation',
    version = '0.0.1',
    author = 'Haoming Chen, Yiting Han, Ivonne Martinez, Gahyun Sung',
    author_email = 'ivonne_martinez@g.harvard.edu',
    description = 'Function Automatic Automatic Differentiation'
    long_description=open('docs/milestone1/README.md').read()
    long_description_content_type="text/markdown",
    url = '',
    install_requires = [
        'some-pkg @ git+ssh://git@github.com/cs107-sandbox/cs107-FinalProject',
        ]
    install_requires=['numpy', 'pytest', 'pip'],
    license ='MIT',
    package_dir = {"":"src"},
    packages = setuptools.find_packages(where='src'),
    python_requires=">3.8",

)
