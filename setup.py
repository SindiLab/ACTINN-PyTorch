from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()



setup(
      name="ACTINN",
      version="0.0.2",
      author="A. Ali Heydari",
      author_email="aliheydari@ucdavis.edu",
      description="A PyTorch implementation of ACTINN: automated identification of cell types in single cell RNA sequencing ",
      long_description=readme,
      long_description_content_type="text/markdown",
      license="MIT",
      url="https://github.com/SindiLab/ACTINN-PyTorch",
      download_url="https://github.com/SindiLab/ACTINN-PyTorch",
      packages=find_packages(),
      install_requires=[
                        'tqdm==4.47.0',
                        'numpy==1.18.5',
                        'torch==1.9.1',
                        'scanpy==1.7.0',
                        'tensorboardX==2.1'
                        ],
      classifiers=[
                   "Development Status :: 1 - Beta",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: MIT Software License",
                   "Programming Language :: Python :: 3.6",
                   "Topic :: Scientific/Engineering :: Artificial Intelligence :: Bioinformatics :: Deep Learning"
                   ],
      keywords="Single Cell RNA-seq, Automatic Classification, Neural Networks, Transfer Learning"
      )
