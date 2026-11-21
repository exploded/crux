//! Stage timings on a real frame. Run with:
//! CRUX_BENCH_FILE=<path> cargo test --release -p crux-core --test bench_stages -- --ignored --nocapture

use crux_core::background::Background;
use crux_core::stars::{blur7_inplace, detect};
use crux_core::{analysis, Config};
use std::time::Instant;

#[test]
#[ignore]
fn stage_timings() {
    let Ok(path) = std::env::var("CRUX_BENCH_FILE") else { return };
    let cfg = Config::default();
    let t = Instant::now();
    let img = crux_core::read_image(std::path::Path::new(&path)).unwrap();
    println!("read        {:>6.0} ms", t.elapsed().as_secs_f64() * 1e3);
    let t = Instant::now();
    let bg = Background::estimate(&img, 64);
    println!("background  {:>6.0} ms", t.elapsed().as_secs_f64() * 1e3);
    let t = Instant::now();
    let mut r = Vec::new();
    bg.subtract(&img, &mut r);
    println!("subtract    {:>6.0} ms", t.elapsed().as_secs_f64() * 1e3);
    let t = Instant::now();
    blur7_inplace(&mut r, img.width, img.height);
    println!("blur        {:>6.0} ms", t.elapsed().as_secs_f64() * 1e3);
    let t = Instant::now();
    let d = detect(&img, &bg, &analysis::detect_params(&cfg, img.adu_scale));
    println!("detect all  {:>6.0} ms ({} stars)", t.elapsed().as_secs_f64() * 1e3, d.stars.len());
    let t = Instant::now();
    let mut r = Vec::new();
    bg.subtract(&img, &mut r);
    let tr = crux_core::trails::detect(&r, img.width, img.height, img.adu_scale, &Default::default());
    println!("trails      {:>6.0} ms ({} found)", t.elapsed().as_secs_f64() * 1e3, tr.len());
    let t = Instant::now();
    let _ = bg.plane();
    println!("plane       {:>6.0} ms", t.elapsed().as_secs_f64() * 1e3);
    let t = Instant::now();
    let m = analysis::analyze_light(&img, &cfg);
    println!("analyse     {:>6.0} ms ({} clean)", t.elapsed().as_secs_f64() * 1e3, m.n_stars);
}
