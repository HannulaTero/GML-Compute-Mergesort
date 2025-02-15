/// @desc CONTROLS.


// Scroll slices.
y += (mouse_wheel_up() - mouse_wheel_down()) * 3 * 16;


// Reset inputs.
if (keyboard_check_pressed(ord("R")))
{
  control.Randomize();
}


// Call mergesort.
if (keyboard_check(ord("1"))) compute.Execute(compute.mergesortV1);
if (keyboard_check(ord("2"))) compute.Execute(compute.mergesortV2);
if (keyboard_check(ord("3"))) compute.Execute(compute.mergesortV3);
if (keyboard_check(ord("4"))) compute.Execute(compute.mergesortV4);


// Call GML sort.
if (keyboard_check(ord("0")))
{
  control.Log("Starting GML sequential sort...");
  control.Log("Moving data from buffer to array first...");
  
  
  control.TimerBegin("Moved inputs to array");
  var _array = array_create(itemCount);
  var _buffInput = buffer.input;
  var _dtype = buffer.dtype;
  buffer_seek(_buffInput, buffer_seek_start, 0);
  for(var i = 0; i < itemCount; i++)
  {
    _array[i] = buffer_read(_buffInput, _dtype);
  }
  control.TimerEnd();
  
  
  control.TimerBegin("Sorted the array");
  array_sort(_array, true);
  control.TimerEnd();
  
  
  control.TimerBegin("Moved outpts to buffer");
  var _buffOutput = buffer.output;
  buffer_seek(_buffOutput, buffer_seek_start, 0);
  for(var i = 0; i < itemCount; i++)
  {
    buffer_write(_buffOutput, _dtype, _array[i]);
  }
  array_resize(_array, 0);
  control.TimerEnd();
  control.Slice();
}
