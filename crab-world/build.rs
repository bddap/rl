fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let lock_path = std::path::Path::new(&manifest)
        .parent()
        .unwrap()
        .join("Cargo.lock");
    println!("cargo:rerun-if-changed={}", lock_path.display());
    let lock = std::fs::read_to_string(lock_path).unwrap();
    let rapier = lock
        .split("[[package]]")
        .find(|p| p.lines().any(|l| l == r#"name = "rapier3d""#))
        .expect("resolved rapier3d pin");
    let pin = rapier
        .lines()
        .find_map(|l| l.strip_prefix("source = "))
        .expect("rapier3d source");
    assert!(
        pin.contains("git+") && pin.contains('#'),
        "Rapier must have a resolved fork pin"
    );
    println!("cargo:rustc-env=RL_RAPIER_PIN={pin}");
}
