/*
  
  This implementation is similar to previous implementation, but it also tries optimize first passes,
  which can be done within shared workgroup memory.

*/
/// @func ComputeMergesortV4(_params);
/// @desc Does parallel mergesort with compute shaders.
/// @param {Struct} _params Check constructor for accepted parameters.
function ComputeMergesortV4(_params={}) constructor
{
  /// @func Dispatch(_params);
  /// @desc Does parallel mergesort with compute shaders. 
  /// @param {Struct} _params Check constructor for accepted parameters.
  /// @returns {Struct.GPUBuffer} Output is created 
  static Dispatch = function(_params)
  {
    // Read parameters.
    var _src = _params[$ "src"]; // Input storage buffer.
    var _dst = _params[$ "dst"]; // Output storage buffer.
    var _aux = _params[$ "aux"]; // Helper storage buffer, auxillary.
    var _stage = _params[$ "stage"]; // Staging buffer to map for reading.
    var _itemOffset = _params[$ "offset"];
    var _itemCount = _params[$ "count"];
    var _callback = _params[$ "callback"];
    
    // Preparations.
    var _auxTemporary = false;
    var _alignment = 256;
    var _passCount = ceil(log2(_itemCount));
    var _byteCount = _itemCount * self.dsize;
    var _byteOffset = _itemOffset * self.dsize;
    var _workgroupCount = ceil(_itemCount * 0.5 / self.workgroupSize);
    
    // Output is required.
    if (_src == undefined)
    {
      throw("Expected source!");
    }
    
    // Output must be power of 2.
    _itemOffset ??= 0;
    _itemCount ??= ceil(_src.size / self.dsize) - _itemOffset;
    if (_itemCount != power(2, _passCount))
    {
      throw("Input count needs to be power of 2!");
    }
    if (_itemCount > (1 << 31))
    {
      throw("Input has too many items, merge can't be represented in i32.");
    }
    
    // Create output, if it's not provided.
    _dst ??= self.device.createBuffer({
      label: "Mergesort Output",
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      size: _byteCount, 
    });
    
    // Mergesort requires auxillary buffer. Basically for ping-ponging.
    if (_aux == undefined)
    {
      _auxTemporary = true;
      _aux = self.device.createBuffer({
        label: "Mergesort Auxillary",
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        size: _byteCount, 
      });
    }
    
    // Create bindgroups for the dispatch.
    var _bindGroups = [
      self.device.createBindGroup({
        label: "Mergesort Bindgroup Global[0]",
        layout: self.bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: self.uniformBuffer, size: 4 * 4 } },
          { binding: 1, resource: { buffer: _aux } },
          { binding: 2, resource: { buffer: _dst } },
        ]
      }),
      self.device.createBindGroup({
        label: "Mergesort Bindgroup Global[1]",
        layout: self.bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: self.uniformBuffer, size: 4 * 4 } },
          { binding: 1, resource: { buffer: _dst } },
          { binding: 2, resource: { buffer: _aux } },
        ]
      })
    ];
    
    
    // Start command encoder, which all operations use.
    var _encoder = self.device.createCommandEncoder({
      label: "Mergesort Command Encoder"
    });
    
    // Copy contents of input into the auxillary.
    // This is optional, as input may be used as auxillary.
    if (_src != _aux)
    {
      _encoder.copyBufferToBuffer(_src, _byteOffset, _aux, 0, _byteCount);
    }
    
    // Begin the compute pass.
    var _pass = _encoder.beginComputePass();
    _pass.setBindGroup(0, _bindGroups[0], [ 0 ]);
    _pass.setPipeline(self.pipelineComputeWorkgroup);
    _pass.dispatchWorkgroups(_workgroupCount);
    var _passesDone = ceil(log2(self.workgroupSize * 2));
    
    // Do the compute passes.
    // Each invocation handles two values, invocation count therefore is (itemCount/2).
    _pass.setPipeline(self.pipelineComputeGlobal);
    for(var i = _passesDone; i < _passCount; i++)
    {
      _pass.setBindGroup(0, _bindGroups[i % 2], [ i * _alignment ]);
      _pass.dispatchWorkgroups(_workgroupCount);
    }
    _pass.end_();
    
    
    // Make sure destination is updated as the last.
    if ((_passCount % 2) == 0)
    {
      _encoder.copyBufferToBuffer(_aux, 0, _dst, 0, _byteCount);
    }
    
    // Move results to staging buffer, which can be used to copy to GML buffer.
    if (_stage != undefined)
    {
      _encoder.copyBufferToBuffer(_dst, 0, _stage, 0, _byteCount);
    }
    
    // Submit for the execution.
    var _commandBuffer = _encoder.finish();
    self.device.queue.submit([_commandBuffer]);
    if (_callback != undefined)
    {
      self.device.queue.onSubmittedWorkDone(_callback);
    }
    
    // Destroy auxillary.
    if (_auxTemporary == true)
    {
      _aux.destroy();
    }
    
    return _dst;
  }

  
  // Define what data the buffer contains.
  self.device = _params[$ "device"] ?? GPU.requestAdapter().requestDevice();
  self.dtype = _params[$ "dtype"] ?? "f32";
  self.dsize = _params[$ "dsize"] ?? 4;
  self.workgroupSize = _params[$ "workgroupSize"] ?? 256;
  
  // Require workgroups to be powers of 2.
  if (self.workgroupSize < 1)
  || (self.workgroupSize > 256)
  || (self.workgroupSize != power(2, ceil(log2(self.workgroupSize))))
  {
    throw("Workgroup size should be power of 2, and in range [1] to [256]");
  }
  
  
  // For specifying items as optional struct and how to access specific element.
  var _imports = string_join_ext("\n", _params[$ "import"] ?? []);
  var _access = _params[$ "access"] ?? "";
  
    
  // Create the shader modules.
  self.shaderModule = device.createShaderModule({ 
    label: "Mergesort Shader Module",
    code: self.Replace({
        WORKGROUP_SIZE: string(workgroupSize),
        IMPORTS : _imports,
        DATATYPE : self.dtype,
        ACCESS : _access,
      }, self.shaderSource
    )
  });
  
  
  // Create bindgroup layouts.
  self.bindGroupLayout = device.createBindGroupLayout({
    label: "Mergesort BindGroup Layout",
    entries: [{
      binding: 0, 
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type : "uniform", minBindingSize: 4 * 4, hasDynamicOffset: true },
    }, {
      binding: 1, 
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type : "read-only-storage" },
    }, {
      binding: 2, 
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type : "storage" },
    }]
  });
  
  
  // Create pipeline layout.
  self.pipelineLayout = device.createPipelineLayout({
    label: "Mergesort Pipeline Layout",
    bindGroupLayouts: [ self.bindGroupLayout ]
  });
  
  
  // Create compute pipeline.
  self.pipelineComputeWorkgroup = device.createComputePipeline({
    label: "Mergesort Pipeline",
    layout: self.pipelineLayout,
    compute: {
      module: self.shaderModule,
      entryPoint: "computeMergeWorkgroup"
    }
  });
  
  
  // Create compute pipeline.
  self.pipelineComputeGlobal = device.createComputePipeline({
    label: "Mergesort Pipeline",
    layout: self.pipelineLayout,
    compute: {
      module: self.shaderModule,
      entryPoint: "computeMergeGlobal"
    }
  });
  
    
  // Create uniform containing pass-information for all possible passes.
  // This way information lives in GPU, and only require change dynamic offset.
  var _alignment = 256;
  var _maxPasses = 31;
  var _buffBytes = _maxPasses * _alignment;
  var _buffer = buffer_create(_buffBytes, buffer_fixed, 1);
  for(var i = 0; i < _maxPasses; i++)
  {
    buffer_seek(_buffer, buffer_seek_start, i * _alignment);
    buffer_write(_buffer, buffer_u32, (i + 1));
    buffer_write(_buffer, buffer_u32, (i + 0));
    buffer_write(_buffer, buffer_s32, power(2, i));
    buffer_write(_buffer, buffer_u32, i);
  }

  self.uniformBuffer = device.createBuffer({
    label: "Mergesort Uniform",
    size: _buffBytes, 
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  self.device.queue.writeBuffer(self.uniformBuffer, 0, _buffer, 0, _buffBytes);
  buffer_delete(_buffer);
  
  
  // Mergesort shader -source.
  static shaderSource = (@'
    
    {{IMPORTS}}
    
    
    struct MergeUniform
    {
      groupShift : u32,
      localShift : u32,
      localSize : i32,
      index : u32,
    };
    
    
    @group(0) @binding(0) var <uniform> uni : MergeUniform;
    @group(0) @binding(1) var <storage, read> srcBuff: array<{{DATATYPE}}>;
    @group(0) @binding(2) var <storage, read_write> dstBuff: array<{{DATATYPE}}>;
    
    
    const workgroupSize = {{WORKGROUP_SIZE}}; // Count of invocations in workgroup.
    const workItemCount = workgroupSize * 2; // Each invocation handles two items.
    var<workgroup> auxBuff : array<{{DATATYPE}}, workItemCount>;
    
    
    
    // Compue shader for workgroup-sized sublists.
    @compute 
    @workgroup_size(workgroupSize)
    fn computeMergeWorkgroup(
      @builtin(global_invocation_id) gid : vec3u,
      @builtin(local_invocation_id) lid : vec3u,
      @builtin(workgroup_id) wid : vec3u,
    )
    {
      // Fill the aux-buffer.
      // There are 2 values for each invocation.
      let invocIndex = i32(lid.x);
      let globalStart = i32(wid.x) * workItemCount;
      auxBuff[invocIndex * 2 + 0] = srcBuff[globalStart + invocIndex * 2 + 0];
      auxBuff[invocIndex * 2 + 1] = srcBuff[globalStart + invocIndex * 2 + 1];
      workgroupBarrier();
      
      // Iterate through as many passes as possible with the workgroup.
      for(var localSize : i32 = 1; localSize <= workgroupSize; localSize *= 2)
      {
        // Preparations.
        let groupIndex : i32 = invocIndex / localSize;
        let groupStart : i32 = groupIndex * localSize * 2;
        let localIndex : i32 = invocIndex - groupIndex * localSize;
        let lhsLower : i32 = groupStart;
        let rhsLower : i32 = groupStart + localSize;
        let lhsIndex : i32 = lhsLower + localIndex;
        let rhsIndex : i32 = rhsLower + localIndex;
        let lhsUpper : i32 = rhsLower - 1;
        let rhsUpper : i32 = lhsUpper + localSize;
        workgroupBarrier();
      
      
        // Binary search position for LHS-value in the RHS-sublist.
        var lhsValue = auxBuff[lhsIndex];
        var lower : i32 = rhsLower;
        var upper : i32 = rhsUpper;
        while(lower <= upper)
        {
          var middle = lower + ((upper - lower) >> 1);
          if (lhsValue{{ACCESS}} > auxBuff[middle]{{ACCESS}})
          {
            lower = middle + 1;
          }
          else
          {
            upper = middle - 1;
          }
        }
        var lhsOffset : i32 = (lower - rhsLower);
        
        
        // Binary search position for RHS-value in the LHS-sublist.
        var rhsValue = auxBuff[rhsIndex];
        lower = lhsLower;
        upper = lhsUpper;
        while(lower <= upper)
        {
          var middle = lower + ((upper - lower) >> 1);
          if (rhsValue{{ACCESS}} >= auxBuff[middle]{{ACCESS}})
          {
            lower = middle + 1;
          }
          else
          {
            upper = middle - 1;
          }
        }
        var rhsOffset : i32 = (lower - lhsLower);
        
      
        // Place the RHS item into found position.
        workgroupBarrier();
        auxBuff[groupStart + localIndex + lhsOffset] = lhsValue;
        auxBuff[groupStart + localIndex + rhsOffset] = rhsValue;
        workgroupBarrier();
      }
      
      // Use aux-buffer to update destination.
      workgroupBarrier();
      dstBuff[globalStart + invocIndex * 2 + 0] = auxBuff[invocIndex * 2 + 0];
      dstBuff[globalStart + invocIndex * 2 + 1] = auxBuff[invocIndex * 2 + 1];
    }
    
    
    
    // Compute shader for global passes.
    @compute 
    @workgroup_size(workgroupSize)
    fn computeMergeGlobal(@builtin(global_invocation_id) id : vec3u)
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
      
      
      // Binary search position for LHS-value in the RHS-sublist.
      var value = srcBuff[lhsIndex];
      var lower = rhsStart;
      var upper = rhsStart + uni.localSize - 1;
      while(lower <= upper)
      {
        var middle = lower + ((upper - lower) >> 1);
        if (value{{ACCESS}} > srcBuff[middle]{{ACCESS}})
        {
          lower = middle + 1;
        }
        else
        {
          upper = middle - 1;
        }
      }
      
      
      // Place the LHS item into found position.
      var offset = lower - rhsStart;
      dstBuff[groupStart + localIndex + offset] = value;
      
      
      // Get range for looking in LHS list.
      // We can use knowledge from previous binary search to prune the range.
      lower = lhsStart + (lower - rhsStart);
      
      if (lower > lhsIndex)
      {
        upper = min(lower - 2, lhsIndex - 1);
        lower = lhsStart;
      }
      else
      {
        upper = lhsStart + uni.localSize - 1;
        lower = max(lower + 1, lhsIndex + 1);
      }
      
      
      // Binary search position for RHS-value in the LHS-sublist.
      value = srcBuff[rhsIndex];
      while(lower <= upper)
      {
        var middle = lower + ((upper - lower) >> 1);
        if (value{{ACCESS}} >= srcBuff[middle]{{ACCESS}})
        {
          lower = middle + 1;
        }
        else
        {
          upper = middle - 1;
        }
      }
      
      
      // Place the RHS item into found position.
      offset = lower - lhsStart;
      dstBuff[groupStart + localIndex + offset] = value;
    }
  ');
  
  
  // Helper function.
  static Replace = function(_placeholders={}, _string="")
  {
    static context = { str : "" };
    static functor = method(context, function(_key, _item)
    {
      // feather ignore GM1041
      str = string_replace_all(str, "{{"+_key+"}}", _item);
    });
    context.str = _string;
    struct_foreach(_placeholders, functor);
    return context.str;
  };
}






