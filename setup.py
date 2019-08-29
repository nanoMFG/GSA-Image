from setuptools import find_packages, setup


setup(
    name='gsaimage',
    version='1.2.0-beta',
    long_description=open('README.md').read(),
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    entry_points={
          'gui_scripts': [
              'gsaimage = gsaimage.__main__:main'
          ]
      }
)
