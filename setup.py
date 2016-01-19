from setuptools import setup, Extension
import numpy as np
from Cython.Distutils import build_ext

requires = ['h5py',
            'pandas',
            'scipy',
            'george',
            'emcee',
            'matplotlib',
            'bokeh',
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

data_files = {'spectral_type': ['data/*'],
              'stellar_data': ['data/*']}

optional_requires = ['astropysics',
                     'pyraf', 'mlpy',
                     'anfft']

setup(name='gullikson-scripts',
      version='0.1.3',
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
      packages=['cross_correlation', 'isochrone_helpers', 'fitters', 
                'utils', 'spectral_type', 'stellar_models', 'stellar_data'],
      
      package_data=data_files,
      setup_requires=['cython', 'numpy>=1.6'],
      cmdclass={'build_ext': build_ext},
      ext_modules=[
          Extension("stellar_models.RotBroad_Fast", ["stellar_models/RotBroad2.pyx"], 
                    include_dirs=[np.get_include()], extra_compile_args=["-O3"]),
          Extension("utils.FittingUtilities", ["utils/FittingUtilities.pyx"],
                    include_dirs=[np.get_include()],
                    extra_compile_args=["-O3", "-funroll-loops"]),
      ],
      install_requires=requires,
      extras_require={'Extra stuff': optional_requires},
      include_package_data=True

)
