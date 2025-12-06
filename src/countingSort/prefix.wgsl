
@group(0) @binding(0) var<storage, read_write> counts: array<u32>; 
@group(0) @binding(1) var<storage, read_write> indices: array<vec2<u32>>; 

@compute @workgroup_size(1)
fn main() {
    var total = 0u;
    let size = arrayLength(&counts);

    for (var i = 0u; i < size; i++) {
        let count = counts[i];
        counts[i] = total;
        indices[i] = vec2u(total, count);
        total += count;
    }
}