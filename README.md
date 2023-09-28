# Spatial Distribution Parallel Computing Implementation
This project demonstrates the parallel computing with CUDA. The program calculates the distance of one point to every other point in a data set and distributes the distances in a histogram. It focuses on implementing a spatial distance histogram (SDH) algorithm using GPU acceleration.

One of the key features of this program is the demonstration between the time comparison of the CPU implementation and the GPU implementation. The GPU implementation will obviously be much faster, but it is still amazing to see just how fast it can be compared to the CPU timings.

This project was initially a project for my course in Parallel Computing, but I believe it is still a great reference point for myself, and possibly others who want to see how effective parallel computing can be.
## Explaination of the program output: 

Firstly, you will notice the arguments 50,000 and 1000. The first argument defines the amount of points in the data set and the second argument defines the bucket width, so 50,000 points and a bucket width of 1000 for this example. 

Secondly, the GPU and CPU timings, which is self-explanatory. This is just how long it took for each implementation to finish calculating the spatial distribution algorithm. 

There are 3 histograms present: CPU, GPU, and then the difference between each implementation. The last histogram is to demonstrate that the GPU and CPU algorithm  creates the  same result. 

### Understanding the structure of the histogram
For each histogram, each row is divided into 5 buckets. The number preceding the ":" on the left of every row represents the starting bucket number for the respective row. At the bottom of each histogram shows the total distance, T.

![image](https://github.com/YingJames/Spatial-Distribution-GPU-Implementation/assets/21976362/b5ab9f72-6f04-4544-b3e1-f36d3288b9b1)

## Goals
The overarching goal is to optimize the program to decrease calculation time, but I'm also interested in the idea of representing this data outside of the terminal output.
- [ ] clean up variable naming and comments
- [ ] take advantage of memory coalescing
- [ ] represent the histogram data in a more visual manner (possibly with a python library?)
