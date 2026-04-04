// gpu.rs — WebGPU device init + compute pipeline helpers

use wgpu::util::DeviceExt;
use wgpu::*;

/// Holds the WebGPU device, queue, and compiled shader modules.
pub struct GpuContext {
    pub device: Device,
    pub queue: Queue,
    pub conv2d_module: ShaderModule,
    pub manipulator_module: ShaderModule,
    pub upsample_module: ShaderModule,
    pub rgba_to_chw_module: ShaderModule,
    pub chw_to_rgba_module: ShaderModule,
}

impl GpuContext {
    pub async fn new() -> Self {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = Backends::BROWSER_WEBGPU;
        let instance = Instance::new(desc);

        log::info!("Requesting WebGPU adapter...");
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("No WebGPU adapter found — is WebGPU enabled in your browser?");

        log::info!("Adapter: {:?}", adapter.get_info());

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("motion-mag-device"),
                required_features: Features::empty(),
                required_limits: Limits::default(),
                memory_hints: MemoryHints::default(),
                trace: wgpu::Trace::Off,
                experimental_features: Default::default(),
            })
            .await
            .expect("Failed to create WebGPU device");

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

        let rgba_to_chw_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("rgba_to_chw"),
            source: ShaderSource::Wgsl(include_str!("shaders/rgba_to_chw.wgsl").into()),
        });

        let chw_to_rgba_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("chw_to_rgba"),
            source: ShaderSource::Wgsl(include_str!("shaders/chw_to_rgba.wgsl").into()),
        });

        Self {
            device,
            queue,
            conv2d_module,
            manipulator_module,
            upsample_module,
            rgba_to_chw_module,
            chw_to_rgba_module,
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

    /// Record a compute dispatch into an existing compute pass.
    /// This avoids per-dispatch encoder/submit overhead by letting the caller
    /// batch many dispatches into a single command encoder submission.
    pub fn record_dispatch<'a>(
        pass: &mut ComputePass<'a>,
        pipeline: &'a ComputePipeline,
        bind_group: &'a BindGroup,
        workgroups: (u32, u32, u32),
    ) {
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, Some(bind_group), &[]);
        pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }

    /// Helper: ceil division for workgroup count.
    pub fn div_ceil(a: u32, b: u32) -> u32 {
        (a + b - 1) / b
    }
}
