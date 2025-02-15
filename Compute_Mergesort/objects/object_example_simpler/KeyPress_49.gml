/// @desc EXAMPLE USE.


// Dispatch the compute shader.
timer.start = get_timer();
device.queue.writeBuffer(input, 0, array);
mergesort.Dispatch({
  src: input,     // Inputs
  aux: input,     // We don't care about inputs, so we reuse the buffer.
  dst: output,    // Stores the output.
  stage: staging, // We want results to CPU.
});


// Map staging buffer for reading.
staging.mapAsync(GPUMapMode.READ, function(_status, _buffer)
{
  // Mapping may fail.
  if (_status != GPUBufferMapAsyncStatus.SUCCESS)
  {
    show_debug_message("MapAsync failed!");
    return;
  }
  
  // Print out a slice from output.
  // Could also read whole data into buffer.
  show_debug_message(string_repeat("=", 64));
  var _mapped = _buffer.getMappedRange();
  for(var i = 0; i < 32; i++)
  {
    var _index = floor(lerp(0, count, i / 32));
    var _value = _mapped.at(_index);
    slice.index[i] = _index;
    slice.value[i] = _value;
    show_debug_message($"output[{_index}] = {_value}");
  }
  _buffer.unmap();
  
  // Stop the timer.
  timer.stop = get_timer();
});





