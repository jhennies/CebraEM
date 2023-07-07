from setuptools import setup

setup(
    name='cebra-em-core',
    version='0.2.0',
    author='jhennies',
    author_email='hennies@embl.de',
    packages=['cebra_em_core'],
    scripts=[
        'bin/install_torch.py',
        'bin/convert_to_bdv.py',
        'bin/normalize_instances.py'
    ],
    # url='',
    # license='',
    # description='',
    # long_description='',
    # install_requires=[]
)
