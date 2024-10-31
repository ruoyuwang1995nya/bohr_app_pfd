import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="voltcraft_app",
    version="0.0.1",
    author="Ruoyu Wang",
    author_email="ruoyuwang1995@gmail.com",
    description="AI-driven battery simulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "pydflow==1.8.95",
        "pymatgen>=2023.8.10",
        'pymatgen-analysis-defects>=2023.8.22',
        "dpdata>=0.2.13",
        "dpdispatcher",
        "matplotlib",
        "seekpath",
        "fpop>=0.0.7",
        "boto3",
        "pfd-kit @ git+https://github.com/ruoyuwang1995nya/pfd-kit.git@v0.1.0#eqq=pfd-kit",
        "dpgen2 @ git+https://github.com/ruoyuwang1995nya/dpgen2.git@v0.0.8#eqq=dpgen2"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10'
    
)