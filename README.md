# WebGPU Foam Tracing (Radiant Foam rendering)
![GIF of rotating voronoi cell, with cell boundaries transparently rendered in white](https://github.com/ValerADHD/foam-tracing/blob/main/assets/voronoi.gif?raw=true)

This repository contains a WebGPU implementation of the "Radiant Foam" Novel View Synthesis method, as described at https://radfoam.github.io/.
This (partial, unfinished) implementation is only based on the method described in the paper and is not fully optimized, so it may not reflect the performance statistics presented by the original authors.

## Additional features
This repository also includes some unfinished compute shader code for generating a Voronoi diagram from a given set of points. The algorithm implemented is a 3D extension of the "Projector Algorithm" described by Daniel Reem in the paper of the same title (https://arxiv.org/abs/1212.1095). 

## Other notes
The code for initializing WebGPU and the associated window extended from https://sotrh.github.io/learn-wgpu/
