from setuptools import setup, find_packages

setup(
    name='DIAGNOSAI',
    version='1.0.0',
    description='A deep learning system to diagnose diseases from chest X-ray images',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Aditya Gambhir',
    author_email='gambhir.aditya19@gmail.com',
    url='https://github.com/Aditya-gam/DiagnosAI',
    packages=find_packages(),
    install_requires=[
        'torch==2.3.1',
        'torchvision==0.18.1',
        'numpy==1.26.4',
        'pandas==2.2.2',
        'matplotlib',
        'Pillow==10.4.0',
        'scikit-learn==1.5.1',
        'scikit-image==0.24.0',
        'tqdm',
        'opencv-python-headless==4.10.0.84'
    ],
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Healthcare Industry',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
    ],
    keywords='machine learning, image processing, chest x-rays, disease diagnosis',
    entry_points={
        'console_scripts': [
            'diagnosai=src.main:main',
        ],
    },
)
