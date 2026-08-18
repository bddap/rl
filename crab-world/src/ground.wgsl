// The ground scaffold (rl#333 seam 3): the ONE `fn fragment` every ground look
// renders through. It builds the shared GroundCtx (rl::ground::art), dispatches
// to the selected look's `art()` module on a LOOK_* shader def pushed by
// `GroundDetail::specialize` (ground.rs), applies the always-on high-frequency
// detail layer (rl::ground::detail — a look modulates it via grain/relief, never
// skips it), then lighting, then the look's additive glow.
//
// The dispatch chain has NO #else: a variant whose def is missing leaves `art`
// unresolved and pipeline creation fails loudly — no silent fallback to Shipped.
// Guarded by `every_look_def_dispatches_in_the_scaffold` (ground.rs) and the
// per-look screenshot sweep, which specializes every variant on real naga.

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    mesh_view_bindings::globals,
}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif

#import rl::ground::detail::{fine_color, relief_normal}
#import rl::ground::art::{GroundCtx, GroundArt}

#ifdef LOOK_SHIPPED
#import rl::ground::looks::shipped::art
#endif
#ifdef LOOK_NIGHT_BLOOM
#import rl::ground::looks::night_bloom::art
#endif
#ifdef LOOK_PATTERNED_GROUND
#import rl::ground::looks::patterned_ground::art
#endif
#ifdef LOOK_WIND_COMBED
#import rl::ground::looks::wind_combed::art
#endif
#ifdef LOOK_CRACKED_LOAM
#import rl::ground::looks::cracked_loam::art
#endif
#ifdef LOOK_WATERSHED
#import rl::ground::looks::watershed::art
#endif

// x: macro, y: meso (look-owned structure gains); z: grain, w: relief
// (scaffold-owned detail gains, modulated by art()'s grain/relief fields).
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> strengths: vec4<f32>;
// The hydrology bake (moisture.rs, rl#323): R wetness, G standing water. World-
// mapped over the whole tile — uv = world_xz / extent + 0.5, the same transform
// TerrainGrid::height uses — so it has no repeat period to spot from altitude.
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var moisture_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var moisture_smp: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(103) var<uniform> moisture_extent: vec4<f32>;
// The look's aesthetic parameter row (GroundLook::params, ground.rs) — a variant
// of a shared shader is a row, never a fork.
@group(#{MATERIAL_BIND_GROUP}) @binding(104) var<uniform> params: array<vec4<f32>, 8>;

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    let wp = in.world_position.xyz;
    let p = in.world_position.xz;
    // Both footprint derivatives taken HERE, in uniform control flow — every fade
    // downstream may be consumed inside a look's branch. The clamp is a degenerate-
    // derivative guard only — it must sit BELOW any real footprint, or it lies to
    // the detail descents: at 1e-4 it capped fw over most of an on-foot down-look
    // (the crab eye is centimeters up, so near ground is 0.02-0.1 mm/px) and froze
    // the descent's octave choice there (rl#390).
    let fw = max(max(fwidth(p.x), fwidth(p.y)), 1e-5);
    let fw_y = max(fwidth(wp.y), 1e-4);
    let base = pbr_input.material.base_color.rgb;
    let n_geo = normalize(in.world_normal);
    let steep = 1.0 - n_geo.y;

    var ctx: GroundCtx;
    ctx.p = p;
    ctx.wp = wp;
    ctx.fw = fw;
    ctx.fw_y = fw_y;
    ctx.n_geo = n_geo;
    ctx.n = pbr_input.N;
    ctx.v = pbr_input.V;
    ctx.base = base;
    // Vegetation from the biome tint's greenness — growth vs mineral ground.
    ctx.veg = clamp((base.g - max(base.r, base.b)) * 6.0, 0.0, 1.0);
    // Snow mirrored from terrain.rs `biome` (SNOWLINE_M on SNOW_HOLD_STEEP).
    ctx.snow = smoothstep(350.0, 650.0, wp.y) * (1.0 - smoothstep(0.18, 0.38, steep));
    ctx.rough = pbr_input.material.perceptual_roughness;
    // The bake is world-mapped, so its uv needs ABSOLUTE world xz: p is anchor-
    // relative (rl#354) and moisture_extent.zw carries the anchor to add back.
    let muv = (wp.xz + moisture_extent.zw) / moisture_extent.xy + 0.5;
    ctx.hydro = textureSampleLevel(moisture_tex, moisture_smp, muv, 0.0);
    ctx.strengths = strengths;
    ctx.time = globals.time;

    let a = art(ctx, params);

    // The always-on detail layer (rl#333 seam 2): the guaranteed floor under
    // every look's art. grain/relief are the look's multiplicative modulation.
    var rgb = a.rgb;
    // max(): the octave stack's extreme negative tail can sum past -1, and a
    // negative multiplier inverts color under lighting — floor at black.
    rgb *= max(1.0 + fine_color(p, fw, strengths.z * a.grain), 0.0);
    pbr_input.material.base_color = vec4(rgb, pbr_input.material.base_color.a);
    pbr_input.material.perceptual_roughness = a.roughness;
    pbr_input.material.emissive = vec4(
        pbr_input.material.emissive.rgb + a.emissive,
        pbr_input.material.emissive.a,
    );
    pbr_input.N = relief_normal(p, fw, a.n, strengths.w * a.relief);

    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = vec4(out.color.rgb + a.glow, out.color.a);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif
    return out;
}
