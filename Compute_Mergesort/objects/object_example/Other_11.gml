/// @desc DEFINE MERGESORT COMPUTE.

compute.adapter = GPU.requestAdapter();
compute.device = compute.adapter.requestDevice();
compute.mergesortV1 = new ComputeMergesortV1({ device : compute.device });
compute.mergesortV2 = new ComputeMergesortV2({ device : compute.device });
compute.mergesortV3 = new ComputeMergesortV3({ device : compute.device });
compute.mergesortV4 = new ComputeMergesortV4({ device : compute.device });
//compute.mergesortV5 = new ComputeMergesortV5({ device : compute.device });


// Create GPU input and output buffers.
compute.input = compute.device.createBuffer({ 
  label: "Mergesort Storage Input",
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  size: buffer.bytes, 
});

compute.auxillary = compute.device.createBuffer({ 
  // Auxillary buffer is helper buffer, which is required for mergesorting
  // but provided function could automatically create temporary one.
  // Also, if you don't need input buffer anymore, that can also be used as auxillary.
  label: "Mergesort Storage Auxillary",
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  size: buffer.bytes,
});

compute.output = compute.device.createBuffer({ 
  label: "Mergesort Storage Output",
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  size: buffer.bytes,
});

// Output can't be read directly, needs intermediate buffer.
compute.staging = compute.device.createBuffer({ 
  label: "Mergesort Staging Buffer",
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  size: buffer.bytes, 
});


// Executing.
compute.Execute = function(_sorter=compute.mergesortV1)
{
  // Check whether staging is already being mapped.
  if (compute.staging.mapState != "unmapped")
  {
    control.Log("Staging buffer is being mapped (in-use).");
    control.Log(" -> Aborting...");
    return;
  }
  
  
  control.Log($"Dispatching Mergesort...");
  
  // Move inputs from CPU to GPU.
  control.TimerBegin("Time of full dispatch");
  control.TimerBegin("Move inputs to GPU");
  compute.device.queue.writeBuffer(compute.input, 0, buffer.input, 0, buffer.bytes);
  control.TimerEnd();
  
  // Compute and get he results.
  control.TimerBegin("Dispatch compute Pipeline");
  _sorter.Dispatch({
    src: compute.input,
    aux: compute.auxillary,
    dst: compute.output,
    stage: compute.staging,
    offset: 0,
    count: buffer.count,
    callback: function()
    {
      show_debug_message("Callback called!");
    }
  });
  control.TimerEnd();
  
  // Read the results.
  control.TimerBegin("Map outputs for reading");
  compute.staging.mapAsync(GPUMapMode.READ, function(_status, _buffer)
  {
    // Have been mapped.
    control.TimerEnd();
      
    // If there has been an error.
    if (_status != GPUBufferMapAsyncStatus.SUCCESS)
    {
      static errorCases = [
        "Instance dropped",
        "Validation error",
        "Unknown error",
        "Device lost",
        "Destroyed before callback",
        "Unmapped before callback",
        "Offset out of range",
        "Size out of range",
        "Undefined case",
      ];
      control.Log($"MapAsync: {errorCases[_status]}.");
      control.TimerEnd();
      control.TimerEnd();
      control.Log($"Dispatching compute Pipeline failed!");
      return;
    }
    control.Log("MapAsync: Success!");
    
    // Get the results into CPU.
    control.TimerBegin("Read the outputs");
    _buffer.getMappedRange().toBuffer(buffer.output, 0, buffer.bytes, 0);
    _buffer.unmap();
    control.TimerEnd();
    control.TimerEnd();
  
    // Get the results.
    control.Log($"Dispatching mergesort finished!");
    control.Slice();
  });
};
