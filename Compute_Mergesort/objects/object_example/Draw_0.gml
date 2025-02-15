/// @desc DRAW TEXT & VISUALIZE FREEZES ETC.

var _x, _y, _count;
 
// Something moving so visually can see if freezes.
var _rate = (current_time / 3.0);
_x = 64 + dsin(_rate) * 16;
_y = 64 + dcos(_rate) * 16;
draw_set_color(c_white);
draw_circle(_x, _y, 8, false);


// Draw instructions.
var _i = 0;
var _h = 20;
_x = 64;
_y = 128;
draw_set_halign(fa_left);
draw_set_valign(fa_top);
draw_set_color(make_color_hsv(32, 92, 255));
draw_text(_x, _y + (_h * _i++), $"item count  : {buffer.count}");
draw_text(_x, _y + (_h * _i++), $"input range : [{inputRange[0]}, {inputRange[1]}]");
draw_text(_x, _y + (_h * _i++), "");
draw_set_color(make_color_hsv(92, 92, 255));
draw_text(_x, _y + (_h * _i++), "Press:");
draw_text(_x, _y + (_h * _i++), "  [R] Reset input & output.");
draw_text(_x, _y + (_h * _i++), "  [0] Dispatch CPU (Sequential sort).");
draw_text(_x, _y + (_h * _i++), "  [1] Dispatch GPU (Parallel - MergeSort V1).");
draw_text(_x, _y + (_h * _i++), "  [2] Dispatch GPU (Parallel - MergeSort V2).");
draw_text(_x, _y + (_h * _i++), "  [3] Dispatch GPU (Parallel - MergeSort V3).");
draw_text(_x, _y + (_h * _i++), "  [4] Dispatch GPU (Parallel - MergeSort V4).");


// Draw the log.
_count = array_length(control.log);
_x = 32;
_y = room_height - 32 - _count * 14;
draw_set_halign(fa_left);
draw_set_valign(fa_bottom);
draw_set_color(make_color_hsv(160, 92, 255));
for(var i = 0; i < _count; i++)
{
  draw_text(_x, _y, control.log[i]);
  _y += 14;
}


// Draw the slices.
_count = array_length(slice.index);
_x = 720;
_y = 96 + y;
draw_set_halign(fa_right);
draw_set_valign(fa_top);
draw_set_color(make_color_hsv(224, 92, 255));
for(var i = 0; i < _count; i++)
{
  _x = 720;
  draw_text(_x, _y, slice.index[i]);
  _x += 192;
  draw_text(_x, _y, slice.input[i]);
  _x += 192;
  draw_text(_x, _y, slice.output[i]);
  _y += 16;
}


draw_set_color(c_white);




