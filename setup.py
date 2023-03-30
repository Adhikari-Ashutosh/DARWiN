from setuptools import setup, find_packages


setup(
    name='DARWiN',
    version='0.0.0',
    license='MIT',
    author="Ashutosh Adhikari,Ananya Datta,Aditya Kothari",
    author_email='email@example.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/Adhikari-Ashutosh/DARWiN',
    keywords='ML NeuroNET',
    install_requires=[
          'numpy',
      ],

)