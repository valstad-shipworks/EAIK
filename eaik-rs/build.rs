use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cpp_root = manifest_dir.join("../CPP");

    // Find Eigen include path
    let eigen_include = env::var("EIGEN3_INCLUDE_DIR").unwrap_or_else(|_| {
        // Try pkg-config
        if let Ok(lib) = pkg_config::probe_library("eigen3") {
            if let Some(p) = lib.include_paths.first() {
                return p.to_string_lossy().into_owned();
            }
        }
        // Fallback paths
        for p in &["/usr/include/eigen3", "/usr/local/include/eigen3"] {
            if std::path::Path::new(p).exists() {
                return p.to_string();
            }
        }
        panic!("Cannot find Eigen3. Set EIGEN3_INCLUDE_DIR or install via your package manager.");
    });

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .warnings(false)
        // Sources
        .file(manifest_dir.join("ffi/eaik_ffi.cpp"))
        .file(cpp_root.join("src/EAIK.cpp"))
        .file(cpp_root.join("src/IK/utils/kinematic_utils.cpp"))
        .file(cpp_root.join("src/IK/1R_IK.cpp"))
        .file(cpp_root.join("src/IK/2R_IK.cpp"))
        .file(cpp_root.join("src/IK/3R_IK.cpp"))
        .file(cpp_root.join("src/IK/4R_IK.cpp"))
        .file(cpp_root.join("src/IK/5R_IK.cpp"))
        .file(cpp_root.join("src/IK/6R_IK.cpp"))
        .file(cpp_root.join("src/utils/kinematic_remodeling.cpp"))
        .file(cpp_root.join("external/ik-geo/cpp/subproblems/sp.cpp"))
        // Includes
        .include(&eigen_include)
        .include(cpp_root.join("src"))
        .include(cpp_root.join("src/IK"))
        .include(cpp_root.join("src/IK/utils"))
        .include(cpp_root.join("src/utils"))
        .include(cpp_root.join("external/ik-geo/cpp/subproblems"))
        .include(manifest_dir.join("ffi"))
        .compile("eaik_ffi");

    // Link C++ stdlib
    let target = env::var("TARGET").unwrap();
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }

    println!("cargo:rerun-if-changed=ffi/eaik_ffi.cpp");
    println!("cargo:rerun-if-changed=ffi/eaik_ffi.h");
}
