from setuptools import setup, find_packages

setup(
    name="lidar_human_pose_estimation",  # Name of your package
    version="1.0.0",  # Version number
    author="Your Name",
    author_email="simone.arreghini@idsia.ch",
    description="A package for lidar-based human pose estimation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="."),  # Automatically find packages in 'src' directory
    package_dir={"": "."},  # Map top-level package to 'src'
    include_package_data=True,  # Include files specified in MANIFEST.in
    install_requires=[
        "h5py",
        "tqdm",
        "dash",
        "ipdb",
        "black",
        "scipy",
        "numpy>2",
        "pillow",
        "PyYAML",
        "plotly",
        "kaleido",
        "matplotlib",
        "torchscan",
        "tensorboard",
        "torch==2.3.1",
        "torchvision==0.18.1",
        "munkres",
        'scikit-learn==1.6.1',
        'pandas==2.2.3',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
