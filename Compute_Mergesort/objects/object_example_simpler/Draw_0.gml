/// @desc DRAW INFO

// Draw controls etc.
draw_set_halign(fa_left);
draw_set_valign(fa_top);
draw_text(32, 32, $"Count : {count}");
draw_text(32, 48, $"Time  : {(timer.stop - timer.start) / 1000} ms");
draw_text(32, 80, $"[R] Randomize input.");
draw_text(32, 96, $"[1] Dispatch GPU mergesort.");


// Draw slice of outputs.
array_foreach(slice.index, function(_index, i)
{
  draw_set_halign(fa_right);
  draw_text(512 - 16, 64 + i * 16, $"output[{_index}]");
  
  draw_set_halign(fa_left);
  draw_text(512 + 16, 64 + i * 16, slice.value[i]);
});