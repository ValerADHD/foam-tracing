use foamtracing::run;

fn main() {
    pollster::block_on(run());
}