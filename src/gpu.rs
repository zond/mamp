// gpu.rs — WebGPU device init + compute/render pipeline helpers

use wasm_bindgen::JsCast;
use wgpu::*;

pub struct GpuContext {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
}

impl GpuContext {
    pub async fn new() -> Self {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = Backends::BROWSER_WEBGPU;
        let instance = Instance::new(desc);

        // Create surface FIRST — the adapter must be requested with
        // compatible_surface so that wgpu picks an adapter that can actually
        // present to this canvas.  Without this, get_capabilities() may
        // return empty formats on some devices (especially mobile Chrome on
        // Android) and rendering silently produces a black canvas.
        let canvas: web_sys::HtmlCanvasElement = web_sys::window()
            .unwrap()
            .document()
            .unwrap()
            .get_element_by_id("output")
            .unwrap()
            .dyn_into()
            .unwrap();
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .expect("Failed to create surface");

        log::info!("Requesting WebGPU adapter (compatible with surface)...");
        let adapter = match instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
        {
            Ok(a) => a,
            Err(e) => {
                log::warn!(
                    "High-performance adapter failed: {:?}, trying low power...",
                    e
                );
                instance
                    .request_adapter(&RequestAdapterOptions {
                        power_preference: PowerPreference::LowPower,
                        compatible_surface: Some(&surface),
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

        // Drop the temporary surface — we'll create the real one after we know the resolution
        drop(surface);

        Self {
            instance,
            adapter,
            device,
            queue,
        }
    }

    pub fn create_buffer(&self, label: &str, size: u64, usage: BufferUsages) -> Buffer {
        self.device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

}
