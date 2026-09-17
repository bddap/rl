use std::{path::Path, process::Command};

#[path = "src/fnv.rs"]
mod fnv;

fn git(root: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .expect("git available for simulation identity");
    assert!(output.status.success(), "git {args:?} failed");
    String::from_utf8(output.stdout).expect("git output is UTF-8")
}

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let root = Path::new(&manifest).parent().unwrap();
    let commit = git(root, &["rev-parse", "HEAD"]);
    println!("cargo:rustc-env=RL_COMMIT={}", commit.trim());
    let rustc = Command::new(std::env::var_os("RUSTC").unwrap())
        .arg("-vV")
        .output()
        .expect("compiler identity");
    assert!(rustc.status.success());
    let mut build = fnv::Fnv::new();
    build.write(&rustc.stdout);
    for key in [
        "TARGET",
        "CARGO_CFG_TARGET_FEATURE",
        "CARGO_ENCODED_RUSTFLAGS",
        "OPT_LEVEL",
        "DEBUG",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
        build.write(key.as_bytes());
        build.write(std::env::var(key).unwrap_or_default().as_bytes());
    }
    println!("cargo:rustc-env=RL_BUILD_DIGEST={:016x}", build.finish());
    for name in ["HEAD", "index", "packed-refs"] {
        let path = git(root, &["rev-parse", "--git-path", name]);
        println!(
            "cargo:rerun-if-changed={}",
            root.join(path.trim()).display()
        );
    }
    let reference = git(root, &["rev-parse", "--symbolic-full-name", "HEAD"]);
    if reference.trim() != "HEAD" {
        let path = git(root, &["rev-parse", "--git-path", reference.trim()]);
        println!(
            "cargo:rerun-if-changed={}",
            root.join(path.trim()).display()
        );
    }
    let mut source = fnv::Fnv::new();
    for input in [
        "Cargo.toml",
        "Cargo.lock",
        ".cargo/config.toml",
        "crab-world/Cargo.toml",
        "crab-world/build.rs",
        "crab-world/src",
    ] {
        println!("cargo:rerun-if-changed={}", root.join(input).display());
        hash_input(root, Path::new(input), &mut source);
    }
    println!("cargo:rustc-env=RL_SOURCE_DIGEST={:016x}", source.finish());
    let lock = std::fs::read_to_string(root.join("Cargo.lock")).unwrap();
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

fn hash_input(root: &Path, relative: &Path, source: &mut fnv::Fnv) {
    let path = root.join(relative);
    if path.is_dir() {
        let mut entries: Vec<_> = std::fs::read_dir(&path)
            .expect("simulation input directory")
            .map(|e| e.unwrap().file_name())
            .collect();
        entries.sort();
        for entry in entries {
            hash_input(root, &relative.join(entry), source);
        }
    } else {
        source.write(relative.to_str().expect("source path is UTF-8").as_bytes());
        source.write(
            &fnv::fnv1a(
                &std::fs::read(&path)
                    .unwrap_or_else(|e| panic!("simulation input {}: {e}", path.display())),
            )
            .to_le_bytes(),
        );
    }
}
