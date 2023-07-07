from setuptools import setup

setup(
    name='cebra-em',
    version='0.2.0',
    author='jhennies',
    author_email='hennies@embl.de',
    packages=['cebra_em'],
    scripts=[
        'bin/init_project.py',
        'bin/run.py',
        'bin/init_gt.py',
        'bin/link_gt.py',
        'bin/log_gt.py',
        'bin/init_segmentation.py'
    ],
    # url='',
    # license='',
    # description='',
    # long_description='',
    install_requires=[
        'snakemake'
    ]
)
