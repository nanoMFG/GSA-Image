from setuptools import find_packages, setup


setup(
    name='GrResQ',
    version='0.1dev',
    long_description=open('README.md').read(),
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    entry_points={
          'gui_scripts': [
              'gsaimage = gsaimage.__main__:main'
          ]
      }
)
