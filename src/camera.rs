use cgmath::*;
use winit::event::*;
use winit::keyboard::{KeyCode, PhysicalKey};

use wgpu::util::*;

pub struct Camera {
    pub eye: cgmath::Point3<f32>,
    target: cgmath::Point3<f32>,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,

    orbit_radius: f32,
    orbit_angles_deg: cgmath::Vector2<f32>,

    nearest_site: u32,

    pub controller: CameraController,
    pub uniform: CameraUniform
}

impl Camera {
    pub fn new(eye: cgmath::Point3<f32>, target: cgmath::Point3<f32>, aspect: f32, fovy: f32, orbit_radius: f32) -> Self {
        Self {
            eye,
            target,
            up: cgmath::Vector3::unit_y(),
            aspect,
            fovy,
            znear: 0.1,
            zfar: 100.0,

            orbit_radius,
            orbit_angles_deg: (0.0, 0.0).into(),

            nearest_site: 0u32,

            controller: CameraController::new(2.0),
            uniform: CameraUniform::new()
        }
    }

    pub fn create_buffer_bind_group(&self, device: &wgpu::Device) -> (wgpu::Buffer, wgpu::BindGroup, wgpu::BindGroupLayout) {
        let uniform_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[self.uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("camera_bind_group_layout"),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("camera_bind_group"),
        });

        (uniform_buffer, bind_group, layout)
    }

    pub fn update_aspect(&mut self, aspect: f32) {
        self.aspect = aspect;
        self.uniform.aspect = aspect;
    }

    fn to_world_matrix(&self) -> cgmath::Matrix4<f32> {
        cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up).inverse_transform().unwrap()
    }

    pub fn orbit_target(&mut self, target: cgmath::Point3<f32>, view_angles_deg: cgmath::Vector2<f32>) {
        self.target = target;
        self.orbit_angles_deg = view_angles_deg;

        self.eye = {
            let rotation_mat = cgmath::Matrix3::from_angle_y(cgmath::Deg(view_angles_deg.y)) * cgmath::Matrix3::from_angle_x(cgmath::Deg(view_angles_deg.x));

            let eye_offset = rotation_mat * (cgmath::Vector3::unit_z() * self.orbit_radius);

            target + eye_offset
        };
    }

    pub fn update(&mut self) {
        self.orbit_target(self.target, self.controller.get_new_angle(self.orbit_angles_deg));
        self.uniform.update_view_proj(self.to_world_matrix(), self.aspect);
    }

    pub fn update_nearest_site(&mut self, site_idx: u32) {
        self.nearest_site = site_idx;
        self.uniform.nearest_site = site_idx;
    }
}

pub struct CameraController {
    speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool
}

impl CameraController {
    fn new(speed: f32) -> Self {
        Self {
            speed,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false
        }
    }

    pub fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(keycode),
                    state,
                    ..
                },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    KeyCode::KeyW | KeyCode::ArrowUp => {
                        self.is_forward_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyA | KeyCode::ArrowLeft => {
                        self.is_left_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyS | KeyCode::ArrowDown => {
                        self.is_backward_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyD | KeyCode::ArrowRight => {
                        self.is_right_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,

        }
    }

    fn get_new_angle(&self, old_orbit_angles_deg: cgmath::Vector2<f32>) -> cgmath::Vector2<f32> {
        let horizontal_net_input = if self.is_left_pressed { 1 } else { 0 } + if self.is_right_pressed { -1 } else { 0 };
        let vertical_net_input = if self.is_forward_pressed { 1 } else { 0 } + if self.is_backward_pressed { -1 } else { 0 };

        let view_angles_deg =
            old_orbit_angles_deg + cgmath::Vector2::new(vertical_net_input as f32 * -self.speed, horizontal_net_input as f32 * self.speed);

        cgmath::Vector2::new(view_angles_deg.x.clamp(-89.9, 89.9), view_angles_deg.y)
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    camera_to_world: [[f32; 4]; 4],
    aspect: f32,
    nearest_site: u32,
    padding: [f32; 2]
}

impl CameraUniform {
    fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            camera_to_world: cgmath::Matrix4::identity().into(),
            aspect: 1f32,
            nearest_site: 0u32,
            padding: (1., 1.).into()
        }
    }

    fn update_view_proj(&mut self, camera_to_world: cgmath::Matrix4<f32>, aspect: f32) {
        self.camera_to_world = camera_to_world.into();
        self.aspect = aspect;
    }
}