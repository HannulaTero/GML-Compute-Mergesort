
y = 0;
itemCount = 1024 * 1024;
inputRange = [ 0.0, 999.0 ];
timeSummation = 0;
timeIteration = 0;



// Define buffers.
buffer = {};
buffer.dtype = buffer_f32;
buffer.dsize = buffer_sizeof(buffer.dtype);
buffer.count = itemCount;
buffer.bytes = buffer.count * buffer.dsize;
buffer.input = buffer_create(buffer.bytes, buffer_fixed, 1);
buffer.output = buffer_create(buffer.bytes, buffer_fixed, 1);


// Declare Control and debug.
// Defined in user event 0.
slice = {};
slice.count = min(256, itemCount);
slice.index = [];
slice.input = [];
slice.output = [];

control = {};
control.log = [];
control.timers = [];
control.Log = undefined;
control.Slice = undefined;
control.Randomize = undefined;
control.TimeBegin = undefined;
control.TimeEnd = undefined;


// Mergesort compute.
// Defined in user event 1.
compute = {}; 


// Define.
event_perform(ev_other, ev_user0);
event_perform(ev_other, ev_user1);


