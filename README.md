# GML-Compute-Mergesort
 Mergesort with compute shaders for GameMaker (GMRT required).
 
 This was my experiments using compute shaders, and trying to implement mergesort.
 There are few small variations, each trying to do some optimization.
 
 One optimization which I came up with was binary search range pruning, 
 one invocation handles two values, one each from merging sublists.
 After invocation has handled placing the value, it has useful information when choosing
 place for the second value. This can be used to reduce the range for the binary search.
 This is dependent on type of the list, but it can more than half average iterations. 
 In the best case second value can skip whole binary search, but I think that's not something that would usually occur. 
 I haven't seen this kind of optimization yet, but I have not dwelled on sorting algorithm optimizations.
 
 Note that the time is total time from passing data from CPU to GPU, and reading results back.
 The performance would be better, if you don't need to move data between CPU and GPU, 
 for example 
 
 This has implementations with GameMaker's WebGPU bindings.
 Note, that at the time of writing, GMRT is still in beta, and both API's are still in experimental stages. 
 
  Read more https://github.com/YoYoGames/GMRT-Beta/tree/main/docs/webgpu
