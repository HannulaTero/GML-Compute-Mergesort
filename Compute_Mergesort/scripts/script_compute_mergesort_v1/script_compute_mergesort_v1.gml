/*
  
  In this implementation each invocation has responsibility to placing single value.
  The merging happens bottom-up approach. The data has "virtual" sublists, which are merged with each pass.
  The global invocation x-index is used to calculate local index within sublist.
  The global invocation y-index tells whether it reads from LHS and searches RHS, or other way around.
  
*/
/// @func ComputeMergesortV1(_params);
/// @desc Does parallel mergesort with compute shaders. 
/// @param {Struct} _params Check constructor for accepted parameters.
function ComputeMergesortV1(_params={}) constructor
{
  /// @func Dispatch(_params);
  /// @desc Does parallel mergesort with compute shaders. 
  /// @param {Struct} _params Check constructor for accepted parameters.
  /// @returns {Struct.GPUBuffer} Output is created 
  static Dispatch = function(_params)
  {
    // Preparations.
    var _src = _params[$ "src"]; // Input storage buffer.
    var _dst = _params[$ "dst"]; // Output storage buffer.
    var _aux = _params[$ "aux"]; // Helper storage buffer, auxillary.
    var _stage = _params[$ "stage"]; // Staging buffer to map for reading.
    var _itemOffset = _params[$ "offset"];
    var _itemCount = _params[$ "count"];
    var _callback = _params[$ "callback"];
    var _auxTemporary = false;
    var _alignment = 256;
    
    // Output is required.
    if (_src == undefined)
    {
      throw("Expected source!");
    }
    
    // Output must be power of 2.
    _itemOffset ??= 0;
    _itemCount ??= ceil(_src.size / self.dsize) - _itemOffset;
    var _passCount = ceil(log2(_itemCount));
    var _byteCount = _itemCount * self.dsize;
    var _byteOffset = _itemOffset * self.dsize;
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
    
    // Mergesort requires auxillary buffer.
    if (_aux == undefined)
    {
      _auxTemporary = true;
      _aux = self.device.createBuffer({
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
      self.device.createBindGroup({
        label: "Mergesort Bindgroup[0]",
        layout: self.pipelineCompute.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: _uni } },
          { binding: 1, resource: { buffer: _aux } },
          { binding: 2, resource: { buffer: _dst } },
        ]
      }),
      self.device.createBindGroup({
        label: "Mergesort Bindgroup[1]",
        layout: self.pipelineCompute.getBindGroupLayout(0),
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
      var _encoder = self.device.createCommandEncoder();
      _encoder.copyBufferToBuffer(_src, _byteOffset, _aux, 0, _byteCount);
      self.device.queue.submit([_encoder.finish()]);
    }
    
    // Do the compute passes.
    // Submitted with each pass to forces synchronization, 
    // which allows uniforms to be updated in time.
    var _workgroupCount = ceil(_itemCount * 0.5 / self.workgroupSize);
    for(var i = 0; i < _passCount; i++)
    {
      // Set the uniforms.
      buffer_write(_uniBuffer, buffer_u32, (i + 1));
      buffer_write(_uniBuffer, buffer_u32, (i + 0));
      buffer_write(_uniBuffer, buffer_s32, power(2, i));
      buffer_write(_uniBuffer, buffer_u32, i);
      self.device.queue.writeBuffer(_uni, 0, _uniBuffer, 0, _uniBytes);
      
      // Dispatch. 
      _encoder = self.device.createCommandEncoder();
      var _pass = _encoder.beginComputePass();
      _pass.setPipeline(self.pipelineCompute);
      _pass.setBindGroup(0, _bindGroups[i % 2]);
      _pass.dispatchWorkgroups(_workgroupCount, 2);
      _pass.end_();
      self.device.queue.submit([_encoder.finish()]);
    }
    
    
    // Make sure destination has the latest results.
    if ((_passCount % 2) == 0)
    {
      _encoder = self.device.createCommandEncoder();
      _encoder.copyBufferToBuffer(_aux, 0, _dst, 0, _byteCount);
      self.device.queue.submit([_encoder.finish()]);
    }
    
    // Move results to staging buffer, which can be used to copy to GML buffer.
    if (_stage != undefined)
    {
      _encoder = self.device.createCommandEncoder();
      _encoder.copyBufferToBuffer(_dst, 0, _stage, 0, _byteCount);
      self.device.queue.submit([_encoder.finish()]);
    }
    
    // Set callback.
    if (_callback != undefined)
    {
      self.device.queue.onSubmittedWorkDone(_callback);
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
  
  
  // Create compute pipeline.
  self.pipelineCompute = device.createComputePipeline({
    label: "Mergesort Pipeline",
    layout: "auto",
    compute: {
      module: self.shaderModule,
      entrypoint: "computeMerge"
    }
  });
  
  
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
    
    
    // Compute shader for global passes.
    @compute 
    @workgroup_size({{WORKGROUP_SIZE}})
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
      var value : {{DATATYPE}};
      
      
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
          if (value{{ACCESS}} > srcBuff[middle]{{ACCESS}})
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
          if (value{{ACCESS}} >= srcBuff[middle]{{ACCESS}})
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






