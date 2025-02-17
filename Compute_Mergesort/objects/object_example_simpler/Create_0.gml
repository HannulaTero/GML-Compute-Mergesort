/// @desc CREATE REQUIRED STRUCTURES.

// For visualing etc,
slice = {};
slice.index = [];
slice.value = [];
timer = { start: 0, stop: 0 };


// Define inputs.
count = 1024 * 1024;
bytes = count * buffer_sizeof(buffer_f32);
array = array_create(count, 0);


// WebGPU bindings and create mergesorter.
adapter = GPU.requestAdapter();
device = adapter.requestDevice();
mergesort = new ComputeMergesortSimpler();


// GPU input buffer. 
input = device.createBuffer({ 
  label: "Mergesort Storage Input",
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  size: bytes, 
});


// GPU output buffer.
output = device.createBuffer({ 
  label: "Mergesort Storage Output",
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  size: bytes,
});


// Output can't be read directly to CPU, needs intermediate buffer.
staging = device.createBuffer({ 
  label: "Mergesort Staging Buffer",
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  size: bytes, 
});




