const EPSILON = 0.000001;

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) vert_pos: vec3<f32>
};


struct CameraUniform {
    to_world: mat4x4<f32>,
    aspect: f32,
    nearest_site: u32
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

const MAX_NEIGHBORS = 16u;

struct Site {
    position: vec4<f32>,
    color: vec4<f32>,
    neighbor_ids: array<u32, MAX_NEIGHBORS>
}
@group(1) @binding(0)
var<storage, read_write> sites: array<Site>;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    let x= f32(in_vertex_index & 1u) * 2.0 - 1.0;
    let y = f32(in_vertex_index & 2u) - 1.0;


    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.vert_pos = out.clip_position.xyz;
    
    return out;
}

//calculates the intersection between a ray and a (directed) plane
fn ray_dplane_intersection_dist(ray_origin: vec4<f32>, ray_direction: vec4<f32>, plane_normal: vec4<f32>, plane_origin: vec4<f32>) -> f32 {
    if(dot(ray_direction, plane_normal) >= EPSILON) { return -1.0; } 
    
    let t = dot(plane_origin - ray_origin, plane_normal) / dot(ray_direction, plane_normal);

    return t;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let right = camera.to_world[0];
    let up= camera.to_world[1]; 
    let forward = -camera.to_world[2];
    
    let ray_direction = normalize(forward * 1.5 + right * in.vert_pos.x * camera.aspect + up * in.vert_pos.y);

    let ray_origin = camera.to_world[3];
    var p = ray_origin;

    var nearest_site_idx = camera.nearest_site;
    var nearest_site = sites[nearest_site_idx];

    var col = vec4<f32>(0.0, 0.0, 0.0, 1.0);

    var entry_normal = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for(var j: u32 = 0u; j < 16u; j++) {
        // var nearest_intersection: vec4<f32>; var nearest_normal: vec4<f32>;
        // let nearest_intersection_idx = ray_nearest_neighbor(nearest_site_idx, p, ray_direction, &nearest_intersection, &nearest_normal);
        // let nearest_intersection_dist = distance(p, nearest_intersection);
        
        var nearest_intersection = vec4(0.0, 0.0, 0.0, 100.0);
        var nearest_intersection_dist = 1000000000.0;
        var nearest_intersection_idx = 0xffffffffu;
        
        var nearest_normal = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        for(var i: u32 = 0u; i < MAX_NEIGHBORS; i++) {
            let neighbor_idx = nearest_site.neighbor_ids[i];
            if(neighbor_idx == 0xffffffffu) { continue; }
            let neighbor = sites[neighbor_idx];
            
            //vector pointing from sites[i] to the nearest site
            let diff = nearest_site.position - neighbor.position;

            //find the point exactly halfway between the two sites, this will be the origin of our bisecting plane
            let bisector_origin = neighbor.position + diff / 2.0;
            let plane_normal = normalize(diff);

            let intersection_dist = ray_dplane_intersection_dist(p, ray_direction, plane_normal, bisector_origin);
            if(intersection_dist < EPSILON) { continue; }

            if(intersection_dist < nearest_intersection_dist) {
                nearest_intersection = p + intersection_dist * ray_direction;
                nearest_intersection_dist = intersection_dist;
                nearest_intersection_idx = neighbor_idx;
                nearest_normal = plane_normal;
            }
        }

        var traversed_distance = 100.0;
        var traversed_site_color =  nearest_site.color;

        if(nearest_intersection_idx != 0xffffffffu) {
            traversed_distance = nearest_intersection_dist;
            //set the "nearest site" to that of the cell we entered
            nearest_site_idx = nearest_intersection_idx;
            nearest_site = sites[nearest_site_idx];

            //set the ray origin to the position just past the intersected cell boundary
            p = nearest_intersection + ray_direction * EPSILON;
        }

        if(traversed_site_color.a < EPSILON) { traversed_site_color = vec4<f32>(1.0, 1.0, 1.0, 5.0); traversed_distance = 0.05; }
        let l= clamp(dot(entry_normal, vec4<f32>(1.0, 0.0, 0.0, 0.0)) + 1.0, 0.0, 1.0);
        traversed_site_color.r *= l; traversed_site_color.g *= l; traversed_site_color.b *= l; 
        entry_normal = nearest_normal;

        col += vec4<f32>(col.a * (1.0 - exp(-traversed_site_color.a * traversed_distance)) * traversed_site_color.rgb, 0.0);
        col.a *= exp(-traversed_site_color.a * traversed_distance);

        if(nearest_intersection_idx == 0xffffffffu || col.a <= .01) { break; }
    }

    return col + vec4<f32>(0.1, 0.2, 0.3, 1.0) * col.a;
}

const bounding_volume = 10.0;
fn ray_nearest_neighbor(
    cell_id: u32, ray_origin: vec4<f32>, ray_direction: vec4<f32>,
    out_plane_intersection: ptr<function, vec4<f32>>, out_plane_normal: ptr<function, vec4<f32>> 
) -> u32 {
    let cell = sites[cell_id];

    var p: vec4<f32> = ray_origin;
    var nearest_cell_id = cell_id;

    while(abs(p.x) < bounding_volume && abs(p.y) < bounding_volume && abs(p.z) < bounding_volume) {
        p += ray_direction;
        nearest_cell_id = point_nearest_cell(p, cell_id);
        if(nearest_cell_id != cell_id) { break; }
        //if the nearest cell after the initial jump is still the original cell, 
        //jump again until it is not (or we exit the reasonable area)
    }
    if(nearest_cell_id == cell_id) {
        //we reached the boundary without leaving the cell, return a plane for the boundary.
        var plane_origin: vec4<f32>;
        if(abs(ray_direction.x) > abs(ray_direction.y)) {
            if(abs(ray_direction.x) > abs(ray_direction.z)) {
                *out_plane_normal = normalize(vec4<f32>(-ray_direction.x, 0.0, 0.0, 0.0));
                plane_origin = vec4<f32>(sign(ray_direction.x) * bounding_volume, 0.0, 0.0, 1.0);
            } else {
                *out_plane_normal = normalize(vec4<f32>(0.0, 0.0, -ray_direction.z, 0.0));
                plane_origin = vec4<f32>(0.0, 0.0, sign(ray_direction.z) * bounding_volume, 1.0);
            }
        } else {
            if(abs(ray_direction.y) > abs(ray_direction.z)) {
                *out_plane_normal = normalize(vec4<f32>(0.0, -ray_direction.y, 0.0, 0.0));
                plane_origin = vec4<f32>(0.0, sign(ray_direction.y) * bounding_volume, 0.0, 1.0);
            } else {
                *out_plane_normal = normalize(vec4<f32>(0.0, 0.0, -ray_direction.z, 0.0));
                plane_origin = vec4<f32>(0.0, 0.0, sign(ray_direction.z) * bounding_volume, 1.0);
            }
        }
        let t = ray_dplane_intersection_dist(ray_origin, ray_direction, *out_plane_normal, plane_origin);
        *out_plane_intersection = ray_origin + ray_direction * t;
        return 0xffffffffu;
    }

    loop {
        let nearest_cell = sites[nearest_cell_id];

        //find the point exactly halfway between the two cell origins, this will be the origin of our bisecting plane
        let diff = cell.position - nearest_cell.position;
        let bisector_origin = nearest_cell.position + diff / 2.0;
        let plane_normal = normalize(diff);

        let intersection_dist = ray_dplane_intersection_dist(ray_origin, ray_direction, plane_normal, bisector_origin);
        if(intersection_dist == 0.0) { break; }
        let intersection = ray_origin + ray_direction * intersection_dist;

        //the intersection will always* be closer to the ray origin than the previous position
        //because it is at the bisecting plane between the cell nearest to the previous position
        //and our original cell
        p = intersection;

        //find the nearest cell to the new point
        let new_nearest_cell_id = point_nearest_cell(p, cell_id);
        
        if(new_nearest_cell_id == nearest_cell_id || new_nearest_cell_id == cell_id) {
            //we found the same nearest cell twice (or the original cell), so moving to the bisecting plane didn't 
            //find any cells closer than the nearest cell. So, we've found the nearest cell to the original.
            *out_plane_intersection = intersection;
            *out_plane_normal = plane_normal;
            
            return nearest_cell_id;
        }
        
        nearest_cell_id = new_nearest_cell_id;
    }
    return 0xffffffffu;
}


fn point_nearest_cell(point: vec4<f32>, guess_cell_idx: u32) -> u32 {
    var nearest_cell_idx = guess_cell_idx;
    let d = point - sites[guess_cell_idx].position;
    var min_dist = dot(d, d) - EPSILON;
    for(var i = 0u; i < arrayLength(&sites); i++) {
        let diff = point - sites[i].position;
        let sqr_dist = dot(diff, diff);
        if(sqr_dist < min_dist) {
            nearest_cell_idx = i;
            min_dist = sqr_dist;
        }
    }
    return nearest_cell_idx;
}