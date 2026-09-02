use anyhow::{Context, Result};

use clap::Parser;
use crab_world::physics::snapshot::PlantSnapshot;
use crab_world::physics::snapshot::{Pose, Shape, Vec3};

#[derive(Parser)]
pub(crate) struct Args {
    #[arg(long, value_name = "FILE")]
    state: std::path::PathBuf,
    #[arg(long, value_name = "FILE.ppm")]
    out: std::path::PathBuf,
    #[arg(long, default_value_t = 720)]
    size: u32,
}

const RGB_BG: [u8; 3] = [255, 255, 255];
const RGB_CARAPACE: [u8; 3] = [200, 200, 200];
const RGB_LINK: [u8; 3] = [90, 90, 90];
const RGB_WORST: [u8; 3] = [220, 30, 30];

enum Prim {
    Capsule(Vec3, Vec3, f32),
    Box(Pose, Vec3),
}

fn prims(shape: &dyn Shape, pose: Pose, out: &mut Vec<Prim>) {
    if let Some(c) = shape.as_capsule() {
        out.push(Prim::Capsule(
            pose.transform_point(c.segment.a),
            pose.transform_point(c.segment.b),
            c.radius,
        ));
    } else if let Some(b) = shape.as_cuboid() {
        out.push(Prim::Box(pose, b.half_extents));
    } else if let Some(b) = shape.as_ball() {
        let c = pose.transform_point(Vec3::ZERO);
        out.push(Prim::Capsule(c, c, b.radius));
    } else if let Some(comp) = shape.as_compound() {
        for (local, sub) in comp.shapes() {
            prims(&**sub, pose * *local, out);
        }
    }
}

struct Canvas {
    w: u32,
    h: u32,
    px: Vec<u8>,
}

impl Canvas {
    fn new(w: u32, h: u32) -> Self {
        let mut px = Vec::with_capacity((w * h * 3) as usize);
        for _ in 0..w * h {
            px.extend_from_slice(&RGB_BG);
        }
        Self { w, h, px }
    }
    fn put(&mut self, x: i32, y: i32, rgb: [u8; 3]) {
        if x < 0 || y < 0 || x >= self.w as i32 || y >= self.h as i32 {
            return;
        }
        let i = ((y as u32 * self.w + x as u32) * 3) as usize;
        self.px[i..i + 3].copy_from_slice(&rgb);
    }
    fn disc(&mut self, cx: f32, cy: f32, r: f32, rgb: [u8; 3]) {
        let r = r.max(1.0);
        let (x0, x1) = ((cx - r).floor() as i32, (cx + r).ceil() as i32);
        let (y0, y1) = ((cy - r).floor() as i32, (cy + r).ceil() as i32);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
                if dx * dx + dy * dy <= r * r {
                    self.put(x, y, rgb);
                }
            }
        }
    }
    fn line(&mut self, a: (f32, f32), b: (f32, f32), rgb: [u8; 3]) {
        let n = ((b.0 - a.0).abs().max((b.1 - a.1).abs()).ceil() as usize).max(1);
        for i in 0..=n {
            let t = i as f32 / n as f32;
            self.disc(a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t, 1.0, rgb);
        }
    }
}

struct View {
    origin_y: f32,
    scale: f32,
    center: Vec3,
    size: f32,
    axis: fn(Vec3) -> (f32, f32),
}

impl View {
    fn map(&self, p: Vec3) -> (f32, f32) {
        let (u, v) = (self.axis)(p - self.center);
        (
            self.size * 0.5 + u * self.scale,
            self.origin_y + self.size * 0.5 - v * self.scale,
        )
    }
}

fn draw(canvas: &mut Canvas, view: &View, prim: &Prim, rgb: [u8; 3]) {
    match prim {
        Prim::Capsule(a, b, r) => {
            let n = ((*b - *a).length() * view.scale).ceil() as usize + 1;
            for i in 0..=n {
                let p = a.lerp(*b, i as f32 / n as f32);
                let (x, y) = view.map(p);
                canvas.disc(x, y, r * view.scale, rgb);
            }
        }
        Prim::Box(pose, half) => {
            let corner = |sx: f32, sy: f32, sz: f32| {
                view.map(pose.transform_point(Vec3::new(sx * half.x, sy * half.y, sz * half.z)))
            };
            let s = [-1.0, 1.0];
            for &a in &s {
                for &b in &s {
                    canvas.line(corner(a, b, -1.0), corner(a, b, 1.0), rgb);
                    canvas.line(corner(a, -1.0, b), corner(a, 1.0, b), rgb);
                    canvas.line(corner(-1.0, a, b), corner(1.0, a, b), rgb);
                }
            }
        }
    }
}

pub(crate) fn run(args: Args) -> Result<()> {
    let snap = PlantSnapshot::load(&args.state).context("load state")?;
    let (worst, worst_pair) = snap.deepest_same_crab_overlap();
    let mut bodies: Vec<(usize, Vec<Prim>)> = Vec::new();
    for (i, h) in snap.parts.iter().enumerate() {
        let mut ps = Vec::new();
        for (_, co) in snap.colliders.iter() {
            if co.parent() == Some(*h) {
                prims(co.shape(), *co.position(), &mut ps);
            }
        }
        bodies.push((i, ps));
    }
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for (_, ps) in &bodies {
        for p in ps {
            let pts: Vec<Vec3> = match p {
                Prim::Capsule(a, b, r) => vec![*a - Vec3::splat(*r), *b + Vec3::splat(*r)],
                Prim::Box(pose, half) => vec![
                    pose.transform_point(Vec3::ZERO) - Vec3::splat(half.length()),
                    pose.transform_point(Vec3::ZERO) + Vec3::splat(half.length()),
                ],
            };
            for q in pts {
                lo = lo.min(q);
                hi = hi.max(q);
            }
        }
    }
    let size = args.size as f32;
    let extent = (hi - lo).max_element().max(0.1);
    let scale = size * 0.9 / extent;
    let center = (lo + hi) * 0.5;
    let views = [
        View {
            origin_y: 0.0,
            scale,
            center,
            size,
            axis: |p| (p.x, -p.z),
        },
        View {
            origin_y: size,
            scale,
            center,
            size,
            axis: |p| (p.x, p.y),
        },
    ];
    let mut canvas = Canvas::new(args.size, args.size * 2);
    for view in &views {
        for (i, ps) in &bodies {
            let rgb = if worst_pair.is_some_and(|(a, b)| a == *i || b == *i) {
                RGB_WORST
            } else if *i == 0 {
                RGB_CARAPACE
            } else {
                RGB_LINK
            };
            for p in ps {
                draw(&mut canvas, view, p, rgb);
            }
        }
    }
    let mut bytes = format!("P6\n{} {}\n255\n", canvas.w, canvas.h).into_bytes();
    bytes.extend_from_slice(&canvas.px);
    std::fs::write(&args.out, bytes).context("write ppm")?;
    println!(
        "overlap-frame: state after tick {} — deepest same-crab overlap {:.1} mm {:?} → {} (top view above, side view below; scale {:.0} px/m)",
        snap.tick,
        worst * 1000.0,
        worst_pair.map(|(a, b)| (snap.part_joint(a), snap.part_joint(b))),
        args.out.display(),
        scale
    );
    Ok(())
}
