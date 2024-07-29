from setuptools import setup, find_namespace_packages

setup(
    name='PySCF4ASE',
    version='0.0.1',
    author='Kang mingi',
    author_email='kangmg@korea.ac.kr',
    description='PySCF Calculator for ASE Interface',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  
    url='https://github.com/kangmg/PySCF4ASE',
    keywords=['chemistry','computational chemistry', 'ase', 'pyscf'],
    #include_package_data=True,
    packages=find_namespace_packages(), 
    install_requires=[
        'ase',
        'torch',
        'pyscf',
        
        # Check your CUDA version via `nvcc --version`

        # 'gpu4pyscf-cuda12x', # for CUDA version 12.x 
        # 'cutensor-cu12' # for CUDA version 12.x

        # 'gpu4pyscf-cuda11x', # for CUDA version 11.x 
        # 'cutensor-cu11' # for CUDA version 11.x
    ],
    classifiers=[ 
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Chemistry'
    ],
    #python_requires='>=3.10.0',
)