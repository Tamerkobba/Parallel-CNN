# Parallelizing Convolutional Neural Networks (CNNs)

## Overview
This project, led by Tamer Kobba, Yousef Idress, and Mohammad Yazbek, focuses on enhancing the performance and scalability of Convolutional Neural Networks (CNNs) by leveraging parallel computing techniques. The goal is to reduce the computational time required for training CNNs, which is crucial for processing large datasets and complex model architectures.

## Project Structure
- **Sequential/**: Contains the sequential implementation of the CNN, which serves as a baseline for performance comparisons.
- **MPI/**: Implements the CNN using the Message Passing Interface to leverage distributed computing environments.
- **OpenMP/**: Utilizes Open Multi-Processing for parallel computation on shared-memory processors.
- **CUDA/**: Employs CUDA C for GPU-accelerated computing to significantly boost performance.

## Parallelization Techniques
- **MPI**: Focuses on data partitioning and process communication to perform computations across multiple nodes.
- **OpenMP**: Applies parallel directives to computationally intensive loops to improve execution speed on multi-core systems.
- **CUDA C**: Optimizes performance by assigning computations to thousands of threads in a GPU, ideal for tasks with high arithmetic intensity.

## Results
The implementation demonstrates significant speed improvements, particularly with CUDA C, where GPU parallelization provides the most substantial performance gains. Each method's efficacy varies depending on the specific hardware and computational load.

## How to Run
later will

## Future Work
Continued development aims to integrate hybrid parallelization techniques and optimize existing implementations to further enhance the scalability and efficiency of CNN training.

## Contributions
Contributions are welcome, particularly in the areas of improving code efficiency, adding new parallelization strategies, or enhancing documentation.

