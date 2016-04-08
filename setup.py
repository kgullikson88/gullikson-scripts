from setuptools import setup, Extension
import numpy as np
from Cython.Distutils import build_ext

requires = ['h5py',
            'pandas',
            'scipy',
            'george',
            'emcee',
            'matplotlib',
            'bokeh==0.10',
            'astropy',
            'corner',
            'scikit-learn',
            'lmfit',
            'scikit-monaco', 
            'statsmodels',
            'pysynphot',
            'cython',
            'pymultinest',
            'seaborn',
            'astroquery',
            'isochrones',
            'configobj'
            ]

data_files = {'kglib.spectral_type': ['data/*'],
              'kglib.stellar_data': ['data/*']}

optional_requires = ['astropysics',
                     'pyraf', 'mlpy',
                     'anfft']

setup(name='kglib',
      version='0.2.0',
      author='Kevin Gullikson',
      author_email='kevin.gullikson@gmail.com',
      url="https://github.com/kgullikson88/gullikson-scripts",
      description='A series of packages for my analysis',
      license='The MIT License: http://www.opensource.org/licenses/mit-license.php',
      classifiers = [
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',],
      packages=['kglib', 'kglib.cross_correlation', 'kglib.isochrone_helpers',
                'kglib.fitters', 'kglib.utils', 'kglib.spectral_type',
                'kglib.stellar_models', 'kglib.stellar_data'],
      package_data=data_files,
      setup_requires=['cython', 'numpy>=1.6'],
      cmdclass={'build_ext': build_ext},
      ext_modules=[
          Extension("kglib.stellar_models.RotBroad_Fast", ["kglib/stellar_models/RotBroad_Fast.c"],
                    include_dirs=[np.get_include()], extra_compile_args=["-O3"]),
          Extension("kglib.utils.FittingUtilities", ["kglib/utils/FittingUtilities.c"],
                    include_dirs=[np.get_include()],
                    extra_compile_args=["-O3", "-funroll-loops"]),
      ],
      install_requires=requires,
      extras_require={'Extra stuff': optional_requires},
      include_package_data=True

)
