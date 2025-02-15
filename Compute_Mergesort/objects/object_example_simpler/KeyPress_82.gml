/// @desc RANDOMIZE INPUT.


timer.start = get_timer();

// Randomize input.
array_map_ext(array, function()
{
  return irandom_range(1_000, 9_999);
});


// Get slice.
for(var i = 0; i < 32; i++)
{
  var _index = floor(lerp(0, count, i / 32));
  var _value = array[_index];
  slice.index[i] = _index;
  slice.value[i] = _value;
  show_debug_message($"output[{_index}] = {_value}");
}


timer.stop = get_timer();
