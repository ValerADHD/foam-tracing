const EPSILON = 0.0001;

const PI = 3.1415926535897;
const HALF_PI = PI / 2.0;
const TWO_PI = PI * 2.0;

const MAX_NEIGHBORS = 16u;
const NO_CELL = 0xffffffffu;

struct Cell {
    position: vec4<f32>,
    color: vec4<f32>,
    neighbor_ids: array<u32, MAX_NEIGHBORS>
}

@group(0) @binding(0)
var<storage, read_write> cells: array<Cell>;

struct Wedge {
    r1: vec4<f32>,
    r2: vec4<f32>,
    r3: vec4<f32>,
}

const tet = array<vec4<f32>, 4>(
    vec4<f32>(0.000000, -0.333333, -0.816993, 0.0) / 0.8823771962653563, 
    vec4<f32>(0.707537, -0.333333, 0.408497, 0.0) / 0.8823771962653563,
    vec4<f32>(-0.707537, -0.333333, 0.408497, 0.0) / 0.8823771962653563,
    vec4<f32>(0.000000, 1.000000, 0.000000, 0.0)
);

const oct = array<vec4<f32>, 6>(
    vec4<f32>(1.0, 0.0, 0.0, 0.0),
    vec4<f32>(-1.0, 0.0, 0.0, 0.0),
    vec4<f32>(0.0, 1.0, 0.0, 0.0),
    vec4<f32>(0.0, -1.0, 0.0, 0.0),
    vec4<f32>(0.0, 0.0, 1.0, 0.0),
    vec4<f32>(0.0, 0.0, -1.0, 0.0),
);


fn calculate_cell_neighbors(cell_id: u32) -> array<u32, MAX_NEIGHBORS> {
    let cell = cells[cell_id];
    var neighbor_set: array<u32, MAX_NEIGHBORS> = EMPTY_SET;

    // for(var i = 0; i < 64; i++) {
    //     for(var j = 0; j < 64; j++) {
    //         let theta = f32(i) / 64.0 * TWO_PI;
    //         let phi = f32(j) / 64.0 * PI;

    //         let st = sin(theta); let ct = cos(theta);
    //         let sp = sin(phi); let cp = cos(phi);

    //         let rd = vec4<f32>(
    //             st * cp, st * sp, ct, 0.0
    //         );

    //         var po: vec4<f32>; var pn: vec4<f32>;
    //         set_insert(&neighbor_set, ray_nearest_neighbor(cell_id, cell.position, rd, &po, &pn));
    //     }
    // }


    var wedge_queue: array<Wedge, 64>;
    var queue_front = 0;
    var queue_back = 1;

    // wedge_queue[0] = Wedge(oct[0], oct[2], oct[4]);
    // wedge_queue[1] = Wedge(oct[1], oct[2], oct[4]);
    // wedge_queue[2] = Wedge(oct[0], oct[3], oct[4]);
    // wedge_queue[3] = Wedge(oct[1], oct[3], oct[4]);
    // wedge_queue[4] = Wedge(oct[0], oct[2], oct[5]);
    // wedge_queue[5] = Wedge(oct[1], oct[2], oct[5]);
    // wedge_queue[6] = Wedge(oct[0], oct[3], oct[5]);
    // wedge_queue[7] = Wedge(oct[1], oct[3], oct[5]);
    // queue_back = 8;

    // initialize the wedge stack with wedges corresponding to the faces of a tetrahedron
    wedge_queue[0] = Wedge(tet[0], tet[1], tet[2]);
    wedge_queue[1] = Wedge(tet[1], tet[3], tet[2]);
    wedge_queue[2] = Wedge(tet[2], tet[3], tet[0]);
    wedge_queue[3] = Wedge(tet[3], tet[1], tet[0]);
    queue_back = 4;

    for(var iter = 0; iter < 100 && queue_front != queue_back; iter++) {
        let wedge = wedge_queue[queue_front];
        queue_front = (queue_front + 1) % 64;

        //initialize references for plane parameters
        var p1_intersection: vec4<f32>; var p1_norm: vec4<f32>;
        var p2_intersection: vec4<f32>; var p2_norm: vec4<f32>;
        var p3_intersection: vec4<f32>; var p3_norm: vec4<f32>;

        let neighbor_1 = ray_nearest_neighbor(
            cell_id, cell.position, wedge.r1, 
            &p1_intersection, &p1_norm
        );
        let neighbor_2 = ray_nearest_neighbor(
            cell_id, cell.position, wedge.r2, 
            &p2_intersection, &p2_norm
        );
        let neighbor_3 = ray_nearest_neighbor(
            cell_id, cell.position, wedge.r3, 
            &p3_intersection, &p3_norm
        );

        //we want the intersections relative to the cell position--it will be useful later
        let t1 = p1_intersection - cell.position;
        let t2 = p2_intersection - cell.position;
        let t3 = p3_intersection - cell.position;

        //This matrix is solely used to invert later (also to check determinant for solutions)--
        //The system of equations Mt = b allows us to solve for the column vector 't', which represents
        //the intersection of the three planes as coefficients of a linear combination of r1, r2, r3
        let m= transpose(mat3x3<f32>(p1_norm.xyz, p2_norm.xyz, p3_norm.xyz)) * mat3x3<f32>(wedge.r1.xyz, wedge.r2.xyz, wedge.r3.xyz);
        let b = vec3<f32>(dot(p1_norm, t1), dot(p2_norm, t2), dot(p3_norm, t3));

        let d = determinant(m);

        //if the determinant of 'm' is zero, the three plane normals are not linearly independent
        //(or the three rays in this wedge are not linearly independent, but we ensure they are)
        //thus, they are either colinear (making the 3 planes parallel) or coplanar 
        //(making the planes intersect in a line or not at all)
        if(abs(d) < EPSILON) {
            //if the cross product of two vectors is zero, they are colinear
            //if the normals of two planes are colinear, the planes are parallel
            let p12_parallel = abs(length(cross(p1_norm.xyz, p2_norm.xyz))) < EPSILON;
            let p23_parallel = abs(length(cross(p2_norm.xyz, p3_norm.xyz))) < EPSILON;
            let p31_parallel = abs(length(cross(p3_norm.xyz, p1_norm.xyz))) < EPSILON;

            if(p12_parallel && p23_parallel) { //all 3 planes are parallel
                //since the each plane must be the closest bisector in a direction, and they're all parallel,
                //2 of the planes must actually be the same one (2 rays found the same bisector)
                if(neighbor_1 == neighbor_2 && neighbor_2 == neighbor_3) {
                    //all rays found the same neighbor, we can stop processing this wedge
                    set_insert(&neighbor_set, neighbor_1);
                    continue;
                } else {
                    cells[cell_id].color = vec4<f32>(0.0, 1.0, 0.0, 100.0); //TODO
                }
            } else if(p12_parallel || p23_parallel || p31_parallel) { //2 planes are parallel, one is not
                //we want to find a new ray aligned "down" the parallel planes, so save the non-parallel normals
                //(n1 is the normal of the parallel planes, n2 is the norm of the non-parallel plane)
                var n1: vec4<f32>; var n2: vec4<f32>; 
                //we want to split this wedge in two by that ray--the two rays that hit parallel planes
                //need to be split into separate wedges (with the new ray in between). To do this with a single
                //constructor, we'll save r1 and r2 as the rays that hit the parallel planes and r3 as the ray that
                //hit the other plane.
                var r1: vec4<f32>; var r2: vec4<f32>; var r3: vec4<f32>;
                if(p12_parallel) { 
                    n1 = p1_norm; n2 = p3_norm; 
                    r1 = wedge.r1; r2 = wedge.r2; r3 = wedge.r3;
                }
                else if(p23_parallel) { 
                    n1 = p1_norm; n2 = p2_norm; 
                    r1 = wedge.r2; r2 = wedge.r3; r3 = wedge.r1;
                }
                else { //p31 parallel 
                    n1 = p1_norm; n2 = p2_norm; 
                    r1 = wedge.r3; r2 = wedge.r1; r3 = wedge.r2; 
                }

                //our new ray needs to be a linear combination of r1 and r2 (from before) perpendicular to
                //the normal vectors of the parallel planes. We can find the normal vector of the plane formed by
                //r1 and r2 using their cross product, then we can cross that with the normal vector of the parallel planes.
                //the cross product of those two normals must be mutually orthogonal to both, so it is *in the plane of r1/r2*
                // *and* in a parallel plane to the other two.
                var r = vec4(normalize(cross(cross(r1.xyz, r2.xyz), n1.xyz)), 0.0);
                //it could possibly be facing the wrong direction, so make sure it faces in the direction of the wedge.
                r = faceForward(-r, r, wedge.r1);
                
                // cells[cell_id].color.b = 1.0; cells[cell_id].color.a = min(cell.color.a, 1.0);

                wedge_queue[queue_back] = Wedge(
                    r1, r3, r
                );
                wedge_queue[(queue_back + 1) % 64] = Wedge(
                    r2, r3, r
                );
                queue_back = (queue_back + 2) % 64;

            } else { //none of the three planes are parallel
                //If they don't intersect at a point, and they aren't parallel, they must form a "tunnel" 
                //(with possibly zero volume) so create a new ray to shoot in the direction of that tunnel
                var r = normalize(cross(p1_norm.xyz, p2_norm.xyz));
                //make sure it's facing in the direction of this wedge
                r = faceForward(-r, r, wedge.r1.xyz);
                
                //split the wedge we just processed in 3 with the new ray
                wedge_queue[queue_back] = Wedge(
                    wedge.r1, wedge.r2, vec4(r, 0.0)
                );
                wedge_queue[(queue_back + 1) % 64] = Wedge(
                    wedge.r2, wedge.r3, vec4(r, 0.0)
                );
                wedge_queue[(queue_back + 2) % 64] = Wedge(
                    wedge.r3, wedge.r1, vec4(r, 0.0)
                );
                queue_back = (queue_back + 3) % 64;
            }
        } else { //the three planes are in a non-degenerate formation, so there is a single intersection point
            //inverse as the transposed adjugate divided by the determinant
            let m_inverse = transpose(mat3x3<f32>(
                cross(m[1].xyz, m[2].xyz), 
                cross(m[2].xyz, m[0].xyz), 
                cross(m[0].xyz, m[1].xyz)
            )) * (1.0 / d);

            //here's that linear combination mentioned before!
            var coeffs = m_inverse * b;
            let intersection = cell.position + wedge.r1 * coeffs.x + wedge.r2 * coeffs.y + wedge.r3 * coeffs.z;

            //get the ray from the cell position directed towards the intersection
            var r = normalize(intersection - cell.position);
            
            //this wedge is defined as all linear combinations of r1, r2, r3 with positive coefficients. 
            //If that is true for the intersection, we must be in the wedge
            if(coeffs.x > -EPSILON && coeffs.y > -EPSILON && coeffs.z > -EPSILON) {
                //we're in the wedge, so we should test if the point is actually in this cell
                let intersection_cell = point_nearest_cell(intersection, cell_id);
                if(intersection_cell == cell_id) { 
                    //the intersection of the 3 planes is *actually*
                    //in the voronoi cell, so we can stop and add the associated sites as neighbors.
                    set_insert(&neighbor_set, neighbor_1);
                    set_insert(&neighbor_set, neighbor_2);
                    set_insert(&neighbor_set, neighbor_3);
                    continue; //no more work needs to be done for this wedge.
                }
            } else {
                //if the coefficients are all negative, negate the ray (and the coefficients) to put it in the wedge
                if(coeffs.x < 0.0 && coeffs.y < 0.0 && coeffs.z < 0.0) { r = -r; coeffs = -coeffs; }
                else {
                    //if the coefficients aren't all negative, then the ray needs to be "clamped" into the wedge.
                    //For any negative coefficient, we'll zero it out (by subtracting it from itself) and adjust the ray
                    //by subtracting the coefficient * the related ray
                    if(coeffs.x < 0.0) { r -= wedge.r1 * coeffs.x; coeffs.x = 0.0; }
                    if(coeffs.y < 0.0) { r -= wedge.r2 * coeffs.y; coeffs.y = 0.0; }
                    if(coeffs.z < 0.0) { r -= wedge.r3 * coeffs.z; coeffs.z = 0.0; }
                }
            }
            
            // if(!verify_in_wedge(wedge, r)) {
            //     if(!verify_in_wedge(wedge, -r)) { cells[cell_id].color += vec4<f32>(0.0, 0.01, 0.0, 0.01); }
            //     else { cells[cell_id].color += vec4<f32>(0.0, 0.0, 0.01, 0.01); }
            // }

            //split the wedge we just processed in 3 with the new ray
            //If we had to clamp the intersection into the wedge, it will be coplanar with two of the rays
            //this would create a wedge with zero volume, so we want to prevent any of those
            if(coeffs.z > EPSILON) {
                wedge_queue[queue_back] = Wedge(
                    wedge.r1, wedge.r2, r
                );
                queue_back = (queue_back + 1) % 64;
            }
            if(coeffs.x > EPSILON) {
                wedge_queue[queue_back] = Wedge(
                    wedge.r2, wedge.r3, r
                );
                queue_back = (queue_back + 1) % 64;
            }
            if(coeffs.y > EPSILON) {
                wedge_queue[queue_back] = Wedge(
                    wedge.r3, wedge.r1, r
                );
                queue_back = (queue_back + 1) % 64;
            }
        }
    }

    set_coalesce(&neighbor_set);
    return neighbor_set;
}

const initial_jump = 1.0;
const bounding_volume = 30.0;
fn ray_nearest_neighbor(
    cell_id: u32, ray_origin: vec4<f32>, ray_direction: vec4<f32>,
    out_plane_intersection: ptr<function, vec4<f32>>, out_plane_normal: ptr<function, vec4<f32>> 
) -> u32 {
    let cell = cells[cell_id];

    var p: vec4<f32> = ray_origin;
    var nearest_cell_id = cell_id;

    while(abs(p.x) < bounding_volume && abs(p.y) < bounding_volume && abs(p.z) < bounding_volume) {
        p += ray_direction * initial_jump;
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
        let t = ray_plane_intersection_dist(ray_origin, ray_direction, *out_plane_normal, plane_origin);
        *out_plane_intersection = ray_origin + ray_direction * t;
        return NO_CELL;
    }

    loop {
        let nearest_cell = cells[nearest_cell_id];

        //find the point exactly halfway between the two cell origins, this will be the origin of our bisecting plane
        let diff = cell.position - nearest_cell.position;
        let bisector_origin = nearest_cell.position + diff / 2.0;
        let plane_normal = normalize(diff);

        let intersection_dist = ray_plane_intersection_dist(ray_origin, ray_direction, plane_normal, bisector_origin);
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
    return NO_CELL;
}

fn verify_in_wedge(w: Wedge, r: vec4<f32>) -> bool {
    let m = mat3x3<f32>(w.r1.xyz, w.r2.xyz, w.r3.xyz);
    let d = determinant(m);
    if(abs(d) < EPSILON) { return true; }
    let m_inverse = transpose(mat3x3<f32>(
        cross(m[1].xyz, m[2].xyz), 
        cross(m[2].xyz, m[0].xyz), 
        cross(m[0].xyz, m[1].xyz)
    )) * (1.0 / d);

    let coeffs = m_inverse * r.xyz;
    return coeffs.x >= -EPSILON && coeffs.y >= -EPSILON && coeffs.z >= -EPSILON;
}

fn confirm_intersection(p1: vec4<f32>, o1: vec4<f32>, p2: vec4<f32>, o2: vec4<f32>, p3: vec4<f32>, o3: vec4<f32>, intersection: vec4<f32>) -> bool { 
    return distance(vec3<f32>(dot(p1, o1), dot(p2, o2), dot(p3, o3)), vec3<f32>(dot(intersection, p1), dot(intersection, p2), dot(intersection, p3))) < EPSILON;
}

fn point_nearest_cell(point: vec4<f32>, guess_cell_idx: u32) -> u32 {
    var nearest_cell_idx = guess_cell_idx;
    let d = point - cells[guess_cell_idx].position;
    var min_dist = dot(d, d) - EPSILON;
    for(var i = 0u; i < arrayLength(&cells); i++) {
        let diff = point - cells[i].position;
        let sqr_dist = dot(diff, diff);
        if(sqr_dist < min_dist) {
            nearest_cell_idx = i;
            min_dist = sqr_dist;
        }
    }
    return nearest_cell_idx;
}

//calculates the intersection between a ray and a (directed) plane
fn ray_plane_intersection_dist(ray_origin: vec4<f32>, ray_direction: vec4<f32>, plane_normal: vec4<f32>, plane_origin: vec4<f32>) -> f32 {
    if(dot(ray_direction, plane_normal) >= -EPSILON) { return 0.0; } 
    
    let t = dot(plane_origin - ray_origin, plane_normal) / dot(ray_direction, plane_normal);

    return t;
}

const EMPTY_SET = array<u32, MAX_NEIGHBORS>(
    NO_CELL, NO_CELL, NO_CELL, NO_CELL,
    NO_CELL, NO_CELL, NO_CELL, NO_CELL,
    NO_CELL, NO_CELL, NO_CELL, NO_CELL,
    NO_CELL, NO_CELL, NO_CELL, NO_CELL,
);

fn set_insert(s: ptr<function, array<u32, MAX_NEIGHBORS>>, id: u32) -> bool {
    var offset = 0u;
    while(offset < MAX_NEIGHBORS) {
        let existing = (*s)[(id + offset) % MAX_NEIGHBORS];
        if(existing == id) { return false; }
        if(existing == NO_CELL) { (*s)[(id + offset) % MAX_NEIGHBORS] = id; return true; }
        offset++;
    }
    return false;
}

fn set_coalesce(s: ptr<function, array<u32, MAX_NEIGHBORS>>) {
    var insert_index = 0;
    for(var i = 0u; i < MAX_NEIGHBORS; i++) {
        let ele = (*s)[i];
        if(ele != NO_CELL) { (*s)[insert_index] = ele; insert_index++; };
    }
}

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if(global_id.x < arrayLength(&cells)) {
        let cell_neighbors = calculate_cell_neighbors(global_id.x);
        cells[global_id.x].neighbor_ids = cell_neighbors;
    }
}