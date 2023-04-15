from setuptools import setup, find_packages

setup(
    name='tflcg',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'diffusers==0.15.0',
        'Pillow==9.5.0',
        'setuptools==67.6.1',
        'transformers==4.28.0'
    ],
    author='Nipun Jindal',
    author_email='jindal.nipun@gmail.com',
    description='Unofficial huggingface/diffusers-based implementation of the paper Training-Free Layout Control with Cross-Attention Guidance',
    url='https://github.com/yourusername/your-repository',
)
