// gpu.rs — WebGPU device init + compute/render pipeline helpers

use wasm_bindgen::JsCast;
use wgpu::util::DeviceExt;
use wgpu::*;

pub struct GpuContext {
    pub instance: Instance,
    pub device: Device,
    pub queue: Queue,
    pub surface: Surface<'static>,
    pub surface_format: TextureFormat,
    pub rgba_to_chw_module: ShaderModule,
    pub chw_to_rgba_module: ShaderModule,
}

impl GpuContext {
    pub async fn new() -> Self {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = Backends::BROWSER_WEBGPU;
        let instance = Instance::new(desc);

        log::info!("Requesting WebGPU adapter...");
        let adapter = match instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            Ok(a) => a,
            Err(e) => {
                log::warn!("High-performance adapter failed: {:?}, trying low power...", e);
                instance
                    .request_adapter(&RequestAdapterOptions {
                        power_preference: PowerPreference::LowPower,
                        compatible_surface: None,
                        force_fallback_adapter: false,
                    })
                    .await
                    .expect("No WebGPU adapter found. Try Chrome with --enable-unsafe-webgpu")
            }
        };

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

        let rgba_to_chw_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("rgba_to_chw"),
            source: ShaderSource::Wgsl(include_str!("shaders/rgba_to_chw.wgsl").into()),
        });

        let chw_to_rgba_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("chw_to_rgba"),
            source: ShaderSource::Wgsl(include_str!("shaders/chw_to_rgba.wgsl").into()),
        });

        // Create surface on the output canvas (must happen once, before any configure)
        let canvas: web_sys::HtmlCanvasElement = web_sys::window().unwrap()
            .document().unwrap()
            .get_element_by_id("output").unwrap()
            .dyn_into().unwrap();
        let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .expect("Failed to create surface");

        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps.formats.first().copied().unwrap_or(TextureFormat::Bgra8Unorm);
        log::info!("Surface format: {:?}, available: {:?}", surface_format, caps.formats);

        Self { instance, device, queue, surface, surface_format, rgba_to_chw_module, chw_to_rgba_module }
    }

    pub fn create_buffer_init(&self, label: &str, data: &[f32], usage: BufferUsages) -> Buffer {
        self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage,
        })
    }

    pub fn create_buffer(&self, label: &str, size: u64, usage: BufferUsages) -> Buffer {
        self.device.create_buffer(&BufferDescriptor {
            label: Some(label), size, usage,
            mapped_at_creation: false,
        })
    }

    pub fn create_uniform<T: bytemuck::Pod>(&self, label: &str, data: &T) -> Buffer {
        self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(data),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        })
    }

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

    pub fn div_ceil(a: u32, b: u32) -> u32 {
        (a + b - 1) / b
    }
}
