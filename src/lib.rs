mod camera;

use cgmath::InnerSpace;
use winit::{
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{WindowBuilder, Window},
};
use wgpu::util::DeviceExt;

use rand::prelude;
use rand::SeedableRng;
use rand::Rng;
use rand::rngs::StdRng;

use std::time::Instant;



#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Site {
    position: [f32; 4],
    color: [f32; 4],
    neighbor_ids: [u32; 16],
}

impl Site {
    fn new(position: cgmath::Point3<f32>, color: [f32; 4]) -> Self {
        Self {
            position: [position.x, position.y, position.z, 1.0],
            color,
            neighbor_ids: [0xffff_ffffu32; 16]
        }
    }
}

struct State<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: &'a Window,
    render_pipeline: wgpu::RenderPipeline,
    compute_pipeline: wgpu::ComputePipeline,

    camera: camera::Camera,
    camera_uniform_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    sites: Vec<Site>,
    sites_buffer: wgpu::Buffer,
    sites_bindgroup: wgpu::BindGroup,
}

impl<'a> State<'a> {
    async fn new(window: &'a Window) -> State<'a> {
        //take the default window size
        let size = window.inner_size();

        //instantiate WGPU with primary backend + defaults
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::default(),
            ..Default::default()
        });

        //create a surface to address the window
        let surface = instance.create_surface(window).unwrap();

        //grab an adapter (asynchronously, required by WGPU) with default power preference for our surface
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },
        ).await.unwrap();

        //create the WGPU Device and Queue using given adapter (also required async)
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                label: None,
                memory_hints: Default::default()
            },
            None, //no trace path
        ).await.unwrap();

        //get information on the type of surface being written to
        let surface_caps = surface.get_capabilities(&adapter);

        println!("GPU: {} | Driver: {}", adapter.get_info().name, adapter.get_info().driver);

        //find a valid format with preference for SRGB
        let surface_format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied().unwrap_or(surface_caps.formats[0]);

        //set up configuration with format + size, everything else default
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[2],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let mut camera = camera::Camera::new(
            (0.0, 0.0, 1.0).into(),
            (0.0, 0.0, 0.0).into(),
            config.width as f32 / config.height as f32,
            45.0,
            3.0
        );

        camera.orbit_target((0.0, 0.0, 0.0).into(), (0.0, 0.0).into());
        camera.update();

        let (camera_uniform_buffer, camera_bind_group, camera_bind_group_layout)
            = camera.create_buffer_bind_group(&device);

        let mut sites: Vec<Site> = vec![Site::new(cgmath::point3(0.0, 0.0, 0.0), [1.0, 0.0, 0.0, 100.0])];

        let num_random_points = 32;
        let r = StdRng::seed_from_u64(0u64);
        let randoms: Vec<f32> = r.random_iter().take(num_random_points * 4).collect();

        for i in 0..num_random_points {
            let (r1, r2, r3, r4) = (randoms[i * 4], randoms[i * 4 + 1], randoms[i * 4 + 2], randoms[i * 4 + 3]);
            let phi = (2.0 * r1 - 1.0).acos() - std::f32::consts::FRAC_PI_2;
            let theta = r2 * std::f32::consts::PI * 2.0;


            let p: cgmath::Vector3<f32> = cgmath::Matrix3::from_angle_y(cgmath::Rad(theta)) 
                * cgmath::Matrix3::from_angle_x(cgmath::Rad(phi)) 
                * cgmath::Vector3::unit_z() * r3.sqrt();
            
            let mut density = 0.0;
            sites.push(Site::new(cgmath::point3(p.x, p.y, p.z), [0.0, 0.0, 0.0, 0.0]));
        }
        
        let sites_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sites Buffer"),
            contents: bytemuck::cast_slice(&sites[..]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let sites_bindgroup_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ]
        });

        let sites_bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sites Bind Group"),
            layout: &sites_bindgroup_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sites_buffer.as_entire_binding(),
                },
            ]
        });

        //include our shader source file & compile it to a shader module
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout, &sites_bindgroup_layout],
            push_constant_ranges: &[]
        });
        
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let compute_shader = device.create_shader_module(wgpu::include_wgsl!("compute.wgsl"));

        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&sites_bindgroup_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Compute Command Encoder") });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Compute Pass"), timestamp_writes: None });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &sites_bindgroup, &[]);
        compute_pass.dispatch_workgroups(sites.len() as u32, 1, 1);

        drop(compute_pass);

        queue.submit([encoder.finish()]);

        Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
            render_pipeline,
            compute_pipeline,

            camera,
            camera_uniform_buffer,
            camera_bind_group,

            sites,
            sites_buffer,
            sites_bindgroup,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.camera.update_aspect(new_size.width as f32 / new_size.height as f32);
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        self.camera.controller.process_events(event)
    }
    
    fn update(&mut self) {
        self.camera.update();

        //find nearest site to camera as cell to start in
        let (mut nearest_site_dist, mut nearest_site_idx) = (10000.0_f32, 0u32);
        for (idx, site) in self.sites.iter().enumerate() {
            let [x, y, z, _] = site.position;
            let dist = (cgmath::Point3::new(x, y, z) - self.camera.eye).magnitude();
            if dist < nearest_site_dist {
                nearest_site_dist = dist;
                nearest_site_idx = idx as u32;
            }
        }
        self.camera.update_nearest_site(nearest_site_idx);

        self.queue.write_buffer(&self.camera_uniform_buffer, 0, bytemuck::cast_slice(&[self.camera.uniform]));
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder")
        });

        
        //clear screen
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1, g: 0.2, b: 0.3, a: 1.0
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None
        });

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        render_pass.set_bind_group(1, &self.sites_bindgroup, &[]);

        //shader automatically will configure the 4 vertices to the corners of the screen,
        //so just tell it to draw 4 vertices (once)
        render_pass.draw(0..4, 0..1);

        drop(render_pass); //release the mutable borrow on the encoder
        
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

pub async fn run() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut state = State::new(&window).await;

    let mut frame_start: Instant = Instant::now();

    event_loop.run(move |event, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == state.window.id() => if !state.input(event) {
            match event {
                WindowEvent::CloseRequested
                | WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            state: ElementState::Pressed,
                            physical_key: PhysicalKey::Code(KeyCode::Escape),
                            ..
                        },
                    ..
                } => control_flow.exit(),
                WindowEvent::Resized(physical_size) => {
                    state.resize(*physical_size);
                }
                WindowEvent::RedrawRequested => {
                    //tell winit we're going to draw another frame
                    state.window().request_redraw();
                    
                    state.update();
                    match state.render() {
                        Ok(_) => {},
                        //if the surface is lost or out of date, we'll call resize to force it to reconfigure
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => state.resize(state.size),
                        Err(wgpu::SurfaceError::OutOfMemory) => { log::error!("OutOfMemory"); control_flow.exit(); },
                        Err(wgpu::SurfaceError::Timeout) => { log::warn!("Surface timeout"); },
                        _ => {}
                    }

                    let frame_time_millis: f64 = { 
                        let elapsed = frame_start.elapsed().as_micros();
                        (elapsed as f64) / 1000f64
                    };

                    let fps = 1000f64 / frame_time_millis;
    
                    state.window().set_title(&format!("frame time: {:.2} ms | {:.2} FPS", frame_time_millis, fps));
                    frame_start = Instant::now();
                }
                _ => {}
            }
        }
        _ => {}
    }).expect("That definitely didn't work!");
}

