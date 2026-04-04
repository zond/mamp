// gpu.rs — WebGPU device init + compute pipeline helpers

use wgpu::*;

/// Holds the WebGPU device, queue, and compiled shader modules.
pub struct GpuContext {
    pub device: Device,
    pub queue: Queue,
    pub conv2d_module: ShaderModule,
    pub manipulator_module: ShaderModule,
    pub upsample_module: ShaderModule,
    pub frame_io_module: ShaderModule,
}

impl GpuContext {
    pub async fn new() -> Self {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::BROWSER_WEBGPU,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("No WebGPU adapter found");

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("motion-mag-device"),
                    required_features: Features::empty(),
                    required_limits: Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let conv2d_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("conv2d"),
            source: ShaderSource::Wgsl(include_str!("shaders/conv2d.wgsl").into()),
        });

        let manipulator_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("manipulator"),
            source: ShaderSource::Wgsl(include_str!("shaders/manipulator.wgsl").into()),
        });

        let upsample_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("upsample"),
            source: ShaderSource::Wgsl(include_str!("shaders/upsample.wgsl").into()),
        });

        let frame_io_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("frame_io"),
            source: ShaderSource::Wgsl(include_str!("shaders/frame_io.wgsl").into()),
        });

        Self {
            device,
            queue,
            conv2d_module,
            manipulator_module,
            upsample_module,
            frame_io_module,
        }
    }

    /// Create a storage buffer initialized with data.
    pub fn create_buffer_init(&self, label: &str, data: &[f32], usage: BufferUsages) -> Buffer {
        let bytes = bytemuck::cast_slice(data);
        self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some(label),
            contents: bytes,
            usage,
        })
    }

    /// Create an empty storage buffer of given byte size.
    pub fn create_buffer(&self, label: &str, size: u64, usage: BufferUsages) -> Buffer {
        self.device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Create a uniform buffer from a bytemuck-able struct.
    pub fn create_uniform<T: bytemuck::Pod>(&self, label: &str, data: &T) -> Buffer {
        self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(data),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        })
    }

    /// Dispatch a compute pipeline with given bind group.
    pub fn dispatch(
        &self,
        pipeline: &ComputePipeline,
        bind_group: &BindGroup,
        workgroups: (u32, u32, u32),
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("dispatch"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Helper: ceil division for workgroup count.
    pub fn div_ceil(a: u32, b: u32) -> u32 {
        (a + b - 1) / b
    }
}
