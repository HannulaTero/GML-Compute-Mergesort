/*
  
  This contrust is a bit simpler, and only allows sorting F32 values.
  Also it doesn't create cached uniform buffer, instead it sends uniforms each pass.
  
  In this implementation each invocation has responsibility to placing single value.
  The merging happens bottom-up approach. The data has "virtual" sublists, which are merged with each pass.
  The global invocation x-index is used to calculate local index within sublist.
  The global invocation y-index tells whether it reads from LHS and searches RHS, or other way around.
  
*/
/// @func ComputeMergesortSimpler(_params);
/// @desc Does parallel mergesort with compute shaders. 
/// @param {Struct} _params Check constructor for accepted parameters.
function ComputeMergesortSimpler(_params={}) constructor
{
  /// @func Dispatch(_params);
  /// @desc Does parallel mergesort with compute shaders. 
  /// @param {Struct} _params Check constructor for accepted parameters.
  /// @returns {Struct.GPUBuffer} Output is created 
  static Dispatch = function(_params)
  {
    // Get parameters.
    var _src = _params[$ "src"]; // Input storage buffer.
    var _dst = _params[$ "dst"]; // Output storage buffer.
    var _aux = _params[$ "aux"]; // Helper storage buffer, auxillary.
    var _stage = _params[$ "stage"]; // Staging buffer to map for reading.
    var _itemOffset = _params[$ "offset"];
    var _itemCount = _params[$ "count"];
    var _callback = _params[$ "callback"];
    
    // Preparations.
    var _auxTemporary = false;
    var _dsize = buffer_sizeof(buffer_f32);
    var _queue = device.queue;
    var _encoder, _pass;
    
    // Output is required.
    if (_src == undefined)
    {
      throw("Expected source!");
    }
    
    // Output must be power of 2.
    _itemOffset ??= 0;
    _itemCount ??= ceil(_src.size / _dsize) - _itemOffset;
    var _passCount = ceil(log2(_itemCount));
    var _byteCount = _itemCount * _dsize;
    var _byteOffset = _itemOffset * _dsize;
    if (_itemCount != power(2, _passCount))
    {
      throw("Input count needs to be power of 2!");
    }
    if (_itemCount > (1 << 31))
    {
      throw("Input has too many items, merge can't be represented in i32.");
    }
    
    
    // Create output, if it's not provided.
    _dst ??= device.createBuffer({
      label: "Mergesort Output",
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      size: _byteCount, 
    });
    
    // Mergesort requires auxillary buffer.
    if (_aux == undefined)
    {
      _auxTemporary = true;
      _aux = device.createBuffer({
        label: "Mergesort Auxillary",
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        size: _byteCount, 
      });
    }
    
    // Create uniform buffer.
    var _uniBytes = 4 * 4;
    var _uniBuffer = buffer_create(_uniBytes, buffer_wrap, 4);
    var _uni = device.createBuffer({
      label: "Mergesort Uniform",
      size: _uniBytes, 
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    
    
    // Create bindgroups for the dispatch.
    var _bindGroups = [
      device.createBindGroup({
        label: "Mergesort Bindgroup[0]",
        layout: pipelineCompute.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: _uni } },
          { binding: 1, resource: { buffer: _aux } },
          { binding: 2, resource: { buffer: _dst } },
        ]
      }),
      device.createBindGroup({
        label: "Mergesort Bindgroup[1]",
        layout: pipelineCompute.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: _uni } },
          { binding: 1, resource: { buffer: _dst } },
          { binding: 2, resource: { buffer: _aux } },
        ]
      })
    ];
    

    // Copy contents of input into the auxillary.
    // This is optional, as input may be used as auxillary.
    if (_src != _aux)
    {
      _encoder = device.createCommandEncoder();
      _encoder.copyBufferToBuffer(_src, _byteOffset, _aux, 0, _byteCount);
      _queue.submit([_encoder.finish()]);
    }
    
    // Do the compute passes.
    // Submitted with each pass to forces synchronization, 
    // which allows uniforms to be updated in time.
    var _workgroupCount = ceil(_itemCount * 0.5 / workgroupSize);
    for(var i = 0; i < _passCount; i++)
    {
      // Set the uniforms.
      buffer_write(_uniBuffer, buffer_u32, (i + 1));
      buffer_write(_uniBuffer, buffer_u32, (i + 0));
      buffer_write(_uniBuffer, buffer_s32, power(2, i));
      buffer_write(_uniBuffer, buffer_u32, i);
      _queue.writeBuffer(_uni, 0, _uniBuffer, 0, _uniBytes);
      
      // Dispatch. 
      _encoder = device.createCommandEncoder();
      _pass = _encoder.beginComputePass();
      _pass.setPipeline(pipelineCompute);
      _pass.setBindGroup(0, _bindGroups[i % 2]);
      _pass.dispatchWorkgroups(_workgroupCount, 2);
      _pass.end_();
      _queue.submit([ _encoder.finish() ]);
    }
    
    
    // Make sure destination has the latest results.
    if ((_passCount % 2) == 0)
    {
      _encoder = device.createCommandEncoder();
      _encoder.copyBufferToBuffer(_aux, 0, _dst, 0, _byteCount);
      _queue.submit([_encoder.finish()]);
    }
    
    // Move results to staging buffer, which can be used to copy to GML buffer.
    if (_stage != undefined)
    {
      _encoder = device.createCommandEncoder();
      _encoder.copyBufferToBuffer(_dst, 0, _stage, 0, _byteCount);
      _queue.submit([_encoder.finish()]);
    }
    
    // Set callback.
    if (_callback != undefined)
    {
      _queue.onSubmittedWorkDone(_callback);
    }
    
    // Destroy auxillary.
    if (_auxTemporary == true)
    {
      _aux.destroy();
    }
    
    // Destroy uniform buffer.
    _uni.destroy();
    buffer_delete(_uniBuffer);
    
    return _dst;
  }

  
  // Use default GameMaker WebGPU device.
  static device = GPU.requestAdapter().requestDevice();
  static workgroupSize = 256;
  
    
  // Create the shader module.
  static shaderModule = device.createShaderModule({ 
    label: "Mergesort Shader Module",
    code: (@'
    
      struct MergeUniform
      {
        groupShift : u32,
        localShift : u32,
        localSize : i32,
        index : u32,
      };
    
    
      @group(0) @binding(0) var <uniform> uni : MergeUniform;
      @group(0) @binding(1) var <storage, read> srcBuff: array<f32>;
      @group(0) @binding(2) var <storage, read_write> dstBuff: array<f32>;
    
    
      // Compute shader for global passes.
      @compute 
      @workgroup_size(256)
      fn computeMerge(@builtin(global_invocation_id) id : vec3u)
      {
        // Preparations.
        let invocIndex = i32(id.x);
        let groupIndex = invocIndex >> uni.localShift;
        let groupStart = groupIndex << uni.groupShift;
        let localIndex = invocIndex - (groupIndex << uni.localShift);
      
        let lhsStart = groupStart;
        let rhsStart = groupStart + uni.localSize;
        let lhsIndex = lhsStart + localIndex;
        let rhsIndex = rhsStart + localIndex;
      
        var offset : i32 = 0;
        var value : f32;
      
      
        // Invocation handles finding LHS-value position RHS-sublist.
        // Uses binary search to find position. If has same value, finds left-most.
        if (id.y == 0)
        {
          value = srcBuff[lhsIndex];
          var lower = rhsStart;
          var upper = rhsStart + uni.localSize - 1;
          while(lower <= upper)
          {
            var middle = lower + ((upper - lower) >> 1);
            if (value > srcBuff[middle])
            {
              lower = middle + 1;
            }
            else
            {
              upper = middle - 1;
            }
          }
          offset = lower - rhsStart;
        } 
      
      
        // Invocation handles finding RHS-value position LHS-sublist.
        // Uses binary search to find position. If has same value, finds right-most.
        else 
        {
          value = srcBuff[rhsIndex];
          var lower = lhsStart;
          var upper = lhsStart + uni.localSize - 1;
          while(lower <= upper)
          {
            var middle = lower + ((upper - lower) >> 1);
            if (value >= srcBuff[middle])
            {
              lower = middle + 1;
            }
            else
            {
              upper = middle - 1;
            }
          }
          offset = lower - lhsStart;
        }
      
      
        // Finally place the value with offsets.
        dstBuff[groupStart + localIndex + offset] = value;
      }
    ')
  });
  
  
  // Create compute pipeline.
  static pipelineCompute = device.createComputePipeline({
    label: "Mergesort Pipeline",
    layout: "auto",
    compute: {
      entryPoint: "computeMerge",
      module: shaderModule,
    }
  });
}






