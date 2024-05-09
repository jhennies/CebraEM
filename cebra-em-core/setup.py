from setuptools import setup

setup(
    name='cebra-em-core',
    version='0.0.3',
    author='jhennies',
    author_email='hennies@embl.de',
    packages=['cebra_em_core'],
    # scripts=[
    #     'bin/install_torch.py',
    #     'bin/convert_to_bdv.py',
    #     'bin/normalize_instances.py'
    # ],
    entry_points={
        'console_scripts': [
            'convert-to-bdv = bin.convert_to_bdv:main',
            'normalize-instances = bin.normalize_instances:main'
        ]
    },
    # url='',
    # license='',
    # description='',
    # long_description='',
    # install_requires=[]
)
