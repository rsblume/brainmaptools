from setuptools import setup

setup(name='brainmaptools',
      version='0.1',
      description='The funniest joke in the world',
      #url='http://github.com/rsblume/',
      author='Rob Blumenfeld',
      author_email='rsblumenfeld@cpp.edu',
      license='MIT',
      packages=['brainmaptools'],
      install_requires=['networkx', 'numpy','pandas', 'sklearn'],
      zip_safe=False)