/// @desc CREATE REQUIRED STRUCTURES.


adapter = GPU.requestAdapter();
device = adapter.requestDevice();
mergesort = new ComputeMergesortV1();


timer = { start: 0, stop: 0 };
count = 1024 * 1024;
bytes = count * buffer_sizeof(buffer_f32);
array = array_create(count, 0);
slice = {};
slice.index = [];
slice.value = [];


input = device.createBuffer({ 
  label: "Mergesort Storage Input",
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  size: bytes, 
});


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