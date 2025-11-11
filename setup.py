from setuptools import setup, find_packages
   
setup(
    name="barpath",
   version="1.0.0",
    description="Offline Weightlifting Technique Analysis",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        'opencv-python>=4.8.0',
        'mediapipe>=0.10.0',
        'ultralytics>=8.0.0',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'matplotlib>=3.7.0',
        'tqdm>=4.65.0',
        'scipy>=1.10.0'
    ],
    extras_require={
        'gui': ['PyQt6>=6.5.0'],  # or 'tkinter' (built-in)
    },
    entry_points={
           'console_scripts': [
               'barpath=cli.barpath_cli:main',
               'barpath-gui=gui.barpath_gui:main',
           ],
       },
       python_requires='>=3.8',
   )