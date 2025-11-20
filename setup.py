from setuptools import setup, find_packages
   
setup(
    name="barpath",
   version="1.0.0",
    description="Offline Weightlifting Technique Analysis",
    author="Ethan Christian",
    packages=find_packages(),
    install_requires=[
        'opencv-python>=4.8.0',
        'mediapipe>=0.10.0',
        'ultralytics>=8.0.0',
        'torchvision>=0.15.0',
        'onnxruntime>=1.15.0',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'matplotlib>=3.7.0',
        'rich>=13.0.0',
        'pycairo>=1.17.0',
        'toga>=0.4.7'
    ],
    extras_require={},
    entry_points={
        'console_scripts': [
            'barpath=barpath.barpath_cli:main',
            'barpath-gui=barpath.barpath_gui:main',
        ],
    },
       python_requires='>=3.8',
   )