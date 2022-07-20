'''
Function:
    setup the pydrawing
Author:
    Charles
微信公众号:
    Charles的皮卡丘
GitHub:
    https://github.com/CharlesPikachu
'''
import pydrawing
from setuptools import setup, find_packages


'''readme'''
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()


'''package data'''
package_data = {}
package_data.update({
    'pydrawing.modules.beautifiers.pencildrawing': ['textures/*'] 
})
package_data.update({
    'pydrawing.modules.beautifiers.beziercurve': ['potrace.exe'] 
})


'''setup'''
setup(
    name=pydrawing.__title__,
    version=pydrawing.__version__,
    description=pydrawing.__description__,
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent'
    ],
    author=pydrawing.__author__,
    url=pydrawing.__url__,
    author_email=pydrawing.__email__,
    license=pydrawing.__license__,
    include_package_data=True,
    package_data=package_data,
    install_requires=[lab.strip('\n') for lab in list(open('requirements.txt', 'r').readlines())],
    zip_safe=True,
    packages=find_packages(),
)