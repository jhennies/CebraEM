from setuptools import setup

setup(
    name='cebra-em',
    version='0.0.3',
    author='jhennies',
    author_email='hennies@embl.de',
    packages=['cebra_em'],
    # scripts=[
    #     'bin/init_project.py',
    #     'bin/run.py',
    #     'bin/init_gt.py',
    #     'bin/link_gt.py',
    #     'bin/log_gt.py',
    #     'bin/init_segmentation.py'
    # ],
    # url='',
    # license='',
    # description='',
    # long_description='',
    entry_points={
        'console_scripts': [
            'init-gt = bin.init_gt:main',
            'init-project = bin.init_project:main',
            'init-segmentation = bin.init_segmentation:main',
            'link-gt = bin.link_gt:main',
            'log-gt = bin.log_gt:main',
            'run = bin.run:main',
            # 'update_mobie_table = bin.update_mobie_table:main'  # Not implemented yet
        ]
    },
    install_requires=[
        'snakemake'
    ]
)
