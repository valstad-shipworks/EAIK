//! Rust bindings for EAIK — Efficient Analytical Inverse Kinematics.
//!
//! Provides analytical inverse kinematics for 1–6R serial manipulators using
//! subproblem decomposition. Robots can be constructed from either
//! Denavit-Hartenberg parameters ([`DhRobot`]) or H/P kinematic vectors
//! ([`HpRobot`]).
//!
//! # Example
//!
//! ```no_run
//! use eaik::{DhRobot, Pose};
//!
//! // Puma 560 DH parameters
//! let pi = std::f64::consts::PI;
//! let alpha = [0.0, -pi / 2.0, 0.0, -pi / 2.0, pi / 2.0, -pi / 2.0];
//! let a = [0.0, 0.0, 0.4318, -0.0203, 0.0, 0.0];
//! let d = [0.0, 0.14909, 0.0, 0.4331, 0.0, 0.0];
//!
//! let robot = DhRobot::new(&alpha, &a, &d).unwrap();
//! assert!(robot.has_known_decomposition());
//!
//! let q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
//! let pose = robot.fk(&q).unwrap();
//! let solutions = robot.ik(&pose).unwrap();
//!
//! for sol in &solutions {
//!     println!("q = {:?}, is_least_squares = {}", sol.joints, sol.is_least_squares);
//! }
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// Raw FFI bindings
// ---------------------------------------------------------------------------

#[allow(non_camel_case_types)]
mod ffi {
    use std::os::raw::c_char;

    #[repr(C)]
    pub struct EaikRobot {
        _opaque: [u8; 0],
    }

    #[repr(C)]
    pub struct EaikIKSolution {
        _opaque: [u8; 0],
    }

    #[repr(C)]
    pub struct EaikMatrix {
        _opaque: [u8; 0],
    }

    unsafe extern "C" {
        // Construction
        pub fn eaik_robot_new_hp(
            h_data: *const f64,
            n_joints: u32,
            p_data: *const f64,
            r6t: *const f64,
            fixed_indices: *const i32,
            fixed_values: *const f64,
            n_fixed: u32,
            is_double_precision: bool,
            err_buf: *mut c_char,
            err_buf_len: u32,
        ) -> *mut EaikRobot;

        pub fn eaik_robot_new_dh(
            dh_alpha: *const f64,
            dh_a: *const f64,
            dh_d: *const f64,
            n_joints: u32,
            r6t: *const f64,
            fixed_indices: *const i32,
            fixed_values: *const f64,
            n_fixed: u32,
            is_double_precision: bool,
            err_buf: *mut c_char,
            err_buf_len: u32,
        ) -> *mut EaikRobot;

        pub fn eaik_robot_free(robot: *mut EaikRobot);

        // IK
        pub fn eaik_robot_ik(
            robot: *const EaikRobot,
            pose: *const f64,
            err_buf: *mut c_char,
            err_buf_len: u32,
        ) -> *mut EaikIKSolution;

        pub fn eaik_ik_solution_free(sol: *mut EaikIKSolution);
        pub fn eaik_ik_solution_num_solutions(sol: *const EaikIKSolution) -> u32;
        pub fn eaik_ik_solution_num_joints(sol: *const EaikIKSolution) -> u32;
        pub fn eaik_ik_solution_get_q(sol: *const EaikIKSolution, idx: u32, out_q: *mut f64);
        pub fn eaik_ik_solution_is_ls(sol: *const EaikIKSolution, idx: u32) -> bool;

        // FK
        pub fn eaik_robot_fwdkin(
            robot: *const EaikRobot,
            q: *const f64,
            n_joints: u32,
            out_pose: *mut f64,
            err_buf: *mut c_char,
            err_buf_len: u32,
        ) -> i32;

        // Queries
        pub fn eaik_robot_is_spherical(robot: *const EaikRobot) -> bool;
        pub fn eaik_robot_has_known_decomposition(robot: *const EaikRobot) -> bool;
        pub fn eaik_robot_get_kinematic_family(
            robot: *const EaikRobot,
            buf: *mut c_char,
            buf_len: u32,
        ) -> i32;

        // Matrix accessors
        pub fn eaik_robot_get_remodeled_h(robot: *const EaikRobot) -> *mut EaikMatrix;
        pub fn eaik_robot_get_remodeled_p(robot: *const EaikRobot) -> *mut EaikMatrix;
        pub fn eaik_robot_get_original_h(robot: *const EaikRobot) -> *mut EaikMatrix;
        pub fn eaik_robot_get_original_p(robot: *const EaikRobot) -> *mut EaikMatrix;

        pub fn eaik_matrix_free(mat: *mut EaikMatrix);
        pub fn eaik_matrix_rows(mat: *const EaikMatrix) -> u32;
        pub fn eaik_matrix_cols(mat: *const EaikMatrix) -> u32;
        pub fn eaik_matrix_data(mat: *const EaikMatrix) -> *const f64;
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error returned by EAIK operations.
#[derive(Debug, Clone)]
pub struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

/// Shorthand result type for EAIK operations.
pub type Result<T> = std::result::Result<T, Error>;

fn read_err_buf(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

// ---------------------------------------------------------------------------
// Pose (4x4 homogeneous transformation)
// ---------------------------------------------------------------------------

/// A 4x4 homogeneous transformation matrix stored in column-major order.
#[derive(Debug, Clone, Copy)]
pub struct Pose {
    /// Column-major 4x4 data. Index as `data[col * 4 + row]`.
    pub data: [f64; 16],
}

impl Pose {
    /// Create a pose from a column-major 4x4 array.
    pub fn from_col_major(data: [f64; 16]) -> Self {
        Self { data }
    }

    /// Create a pose from a row-major 4x4 array (e.g. `[[r00, r01, ...], ...]` flattened).
    pub fn from_row_major(row_major: [f64; 16]) -> Self {
        let mut data = [0.0; 16];
        for r in 0..4 {
            for c in 0..4 {
                data[c * 4 + r] = row_major[r * 4 + c];
            }
        }
        Self { data }
    }

    /// Create identity pose.
    pub fn identity() -> Self {
        let mut data = [0.0; 16];
        data[0] = 1.0;
        data[5] = 1.0;
        data[10] = 1.0;
        data[15] = 1.0;
        Self { data }
    }

    /// Get element at (row, col), zero-indexed.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[col * 4 + row]
    }

    /// Get the 3x1 translation component.
    pub fn translation(&self) -> [f64; 3] {
        [self.get(0, 3), self.get(1, 3), self.get(2, 3)]
    }

    /// Get the 3x3 rotation component as a row-major array.
    pub fn rotation(&self) -> [[f64; 3]; 3] {
        [
            [self.get(0, 0), self.get(0, 1), self.get(0, 2)],
            [self.get(1, 0), self.get(1, 1), self.get(1, 2)],
            [self.get(2, 0), self.get(2, 1), self.get(2, 2)],
        ]
    }

    /// Return the data as a `[[f64; 4]; 4]` in row-major layout.
    pub fn to_row_major_2d(&self) -> [[f64; 4]; 4] {
        let mut out = [[0.0; 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                out[r][c] = self.get(r, c);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Matrix (dynamic-size, returned by accessors)
// ---------------------------------------------------------------------------

/// A dynamically-sized matrix in column-major order.
#[derive(Debug, Clone)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    /// Column-major data. Element (r, c) is at index `c * rows + r`.
    pub data: Vec<f64>,
}

impl Matrix {
    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get element at (row, col).
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[col * self.rows + row]
    }

    /// Return data as a row-major `Vec<Vec<f64>>`.
    pub fn to_row_major_2d(&self) -> Vec<Vec<f64>> {
        (0..self.rows)
            .map(|r| (0..self.cols).map(|c| self.get(r, c)).collect())
            .collect()
    }
}

/// Convert a raw FFI matrix pointer to an owned [`Matrix`], freeing the C++ side.
fn ffi_matrix_to_owned(ptr: *mut ffi::EaikMatrix) -> Matrix {
    unsafe {
        let rows = ffi::eaik_matrix_rows(ptr) as usize;
        let cols = ffi::eaik_matrix_cols(ptr) as usize;
        let data_ptr = ffi::eaik_matrix_data(ptr);
        let data = std::slice::from_raw_parts(data_ptr, rows * cols).to_vec();
        ffi::eaik_matrix_free(ptr);
        Matrix { rows, cols, data }
    }
}

// ---------------------------------------------------------------------------
// IK Solution
// ---------------------------------------------------------------------------

/// A single IK solution: one joint configuration.
#[derive(Debug, Clone)]
pub struct IkSolution {
    /// Joint angles for this solution.
    pub joints: Vec<f64>,
    /// Whether this solution is a least-squares approximation.
    pub is_least_squares: bool,
}

/// All IK solutions for a given pose.
#[derive(Debug, Clone)]
pub struct IkResult {
    /// Individual solutions (may be empty if no solution exists).
    pub solutions: Vec<IkSolution>,
}

impl IkResult {
    /// Number of solutions found.
    pub fn len(&self) -> usize {
        self.solutions.len()
    }

    /// Returns `true` if no solutions were found.
    pub fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }

    /// Iterate over solutions.
    pub fn iter(&self) -> std::slice::Iter<'_, IkSolution> {
        self.solutions.iter()
    }

    /// Get only the exact (non-least-squares) solutions.
    pub fn exact_solutions(&self) -> Vec<&IkSolution> {
        self.solutions
            .iter()
            .filter(|s| !s.is_least_squares)
            .collect()
    }
}

impl<'a> IntoIterator for &'a IkResult {
    type Item = &'a IkSolution;
    type IntoIter = std::slice::Iter<'a, IkSolution>;
    fn into_iter(self) -> Self::IntoIter {
        self.solutions.iter()
    }
}

impl IntoIterator for IkResult {
    type Item = IkSolution;
    type IntoIter = std::vec::IntoIter<IkSolution>;
    fn into_iter(self) -> Self::IntoIter {
        self.solutions.into_iter()
    }
}

/// Extract an [`IkResult`] from a raw FFI solution pointer, freeing the C++ side.
fn ffi_ik_to_result(ptr: *mut ffi::EaikIKSolution) -> IkResult {
    unsafe {
        let n_sol = ffi::eaik_ik_solution_num_solutions(ptr) as usize;
        let n_joints = ffi::eaik_ik_solution_num_joints(ptr) as usize;
        let mut solutions = Vec::with_capacity(n_sol);
        let mut buf = vec![0.0f64; n_joints];
        for i in 0..n_sol {
            ffi::eaik_ik_solution_get_q(ptr, i as u32, buf.as_mut_ptr());
            let is_ls = ffi::eaik_ik_solution_is_ls(ptr, i as u32);
            solutions.push(IkSolution {
                joints: buf.clone(),
                is_least_squares: is_ls,
            });
        }
        ffi::eaik_ik_solution_free(ptr);
        IkResult { solutions }
    }
}

// ---------------------------------------------------------------------------
// Fixed axis specification
// ---------------------------------------------------------------------------

/// A joint axis locked to a fixed angle (for redundant manipulators).
#[derive(Debug, Clone, Copy)]
pub struct FixedAxis {
    /// Zero-indexed joint index.
    pub joint: usize,
    /// Fixed angle value (radians).
    pub angle: f64,
}

impl FixedAxis {
    pub fn new(joint: usize, angle: f64) -> Self {
        Self { joint, angle }
    }
}

// ---------------------------------------------------------------------------
// Shared robot handle
// ---------------------------------------------------------------------------

/// Internal owning wrapper around the C++ robot pointer.
struct RobotHandle {
    ptr: *mut ffi::EaikRobot,
    n_joints: u32,
}

// SAFETY: The C++ Robot is internally thread-safe for const methods.
// Construction is done single-threaded, and all query methods are const.
unsafe impl Send for RobotHandle {}
unsafe impl Sync for RobotHandle {}

impl Drop for RobotHandle {
    fn drop(&mut self) {
        unsafe { ffi::eaik_robot_free(self.ptr) }
    }
}

impl RobotHandle {
    fn ik(&self, pose: &Pose) -> Result<IkResult> {
        let mut err = [0u8; 512];
        let sol = unsafe {
            ffi::eaik_robot_ik(self.ptr, pose.data.as_ptr(), err.as_mut_ptr().cast(), 512)
        };
        if sol.is_null() {
            return Err(Error(read_err_buf(&err)));
        }
        Ok(ffi_ik_to_result(sol))
    }

    fn fk(&self, q: &[f64]) -> Result<Pose> {
        let mut err = [0u8; 512];
        let mut pose = Pose { data: [0.0; 16] };
        let rc = unsafe {
            ffi::eaik_robot_fwdkin(
                self.ptr,
                q.as_ptr(),
                q.len() as u32,
                pose.data.as_mut_ptr(),
                err.as_mut_ptr().cast(),
                512,
            )
        };
        if rc != 0 {
            return Err(Error(read_err_buf(&err)));
        }
        Ok(pose)
    }

    fn is_spherical(&self) -> bool {
        unsafe { ffi::eaik_robot_is_spherical(self.ptr) }
    }

    fn has_known_decomposition(&self) -> bool {
        unsafe { ffi::eaik_robot_has_known_decomposition(self.ptr) }
    }

    fn kinematic_family(&self) -> String {
        let mut buf = [0u8; 256];
        unsafe {
            ffi::eaik_robot_get_kinematic_family(self.ptr, buf.as_mut_ptr().cast(), 256);
        }
        read_err_buf(&buf)
    }

    fn remodeled_h(&self) -> Matrix {
        ffi_matrix_to_owned(unsafe { ffi::eaik_robot_get_remodeled_h(self.ptr) })
    }

    fn remodeled_p(&self) -> Matrix {
        ffi_matrix_to_owned(unsafe { ffi::eaik_robot_get_remodeled_p(self.ptr) })
    }

    fn original_h(&self) -> Matrix {
        ffi_matrix_to_owned(unsafe { ffi::eaik_robot_get_original_h(self.ptr) })
    }

    fn original_p(&self) -> Matrix {
        ffi_matrix_to_owned(unsafe { ffi::eaik_robot_get_original_p(self.ptr) })
    }
}

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

fn parse_fixed_axes(fixed: &[FixedAxis]) -> (Vec<i32>, Vec<f64>) {
    let indices: Vec<i32> = fixed.iter().map(|f| f.joint as i32).collect();
    let values: Vec<f64> = fixed.iter().map(|f| f.angle).collect();
    (indices, values)
}

fn fixed_ptrs(indices: &[i32], values: &[f64]) -> (*const i32, *const f64, u32) {
    let n = indices.len() as u32;
    let ip = if indices.is_empty() {
        std::ptr::null()
    } else {
        indices.as_ptr()
    };
    let vp = if values.is_empty() {
        std::ptr::null()
    } else {
        values.as_ptr()
    };
    (ip, vp, n)
}

// ---------------------------------------------------------------------------
// HpRobot
// ---------------------------------------------------------------------------

/// A robot defined by H (joint axis directions) and P (joint offsets).
///
/// H has shape (3, n_joints) and P has shape (3, n_joints + 1), both in
/// column-major order. Each column of H is a unit vector along the
/// corresponding joint axis; each column of P is the linear offset between
/// consecutive axes.
pub struct HpRobot {
    handle: RobotHandle,
}

impl HpRobot {
    /// Create a new HP robot.
    ///
    /// - `h`: Flattened column-major (3 x n) joint axis matrix.
    /// - `p`: Flattened column-major (3 x (n+1)) joint offset matrix.
    ///
    /// Both slices should have lengths that are multiples of 3.
    pub fn new(h: &[f64], p: &[f64]) -> Result<Self> {
        Self::new_with_options(h, p, None, &[], true)
    }

    /// Create an HP robot with full options.
    ///
    /// - `r6t`: Optional 3x3 column-major end-effector rotation. `None` = identity.
    /// - `fixed_axes`: Joints to lock at fixed angles (for redundant robots).
    /// - `double_precision`: Use double-precision thresholds.
    pub fn new_with_options(
        h: &[f64],
        p: &[f64],
        r6t: Option<&[f64; 9]>,
        fixed_axes: &[FixedAxis],
        double_precision: bool,
    ) -> Result<Self> {
        assert!(h.len() % 3 == 0, "H length must be a multiple of 3");
        assert!(p.len() % 3 == 0, "P length must be a multiple of 3");
        let n_joints = (h.len() / 3) as u32;
        assert_eq!(
            p.len(),
            3 * (n_joints as usize + 1),
            "P must have 3*(n_joints+1) elements"
        );

        let (fi, fv) = parse_fixed_axes(fixed_axes);
        let (fi_ptr, fv_ptr, nf) = fixed_ptrs(&fi, &fv);
        let r6t_ptr = r6t.map_or(std::ptr::null(), |r| r.as_ptr());

        let mut err = [0u8; 512];
        let ptr = unsafe {
            ffi::eaik_robot_new_hp(
                h.as_ptr(),
                n_joints,
                p.as_ptr(),
                r6t_ptr,
                fi_ptr,
                fv_ptr,
                nf,
                double_precision,
                err.as_mut_ptr().cast(),
                512,
            )
        };
        if ptr.is_null() {
            return Err(Error(read_err_buf(&err)));
        }
        Ok(Self {
            handle: RobotHandle { ptr, n_joints },
        })
    }

    /// Compute inverse kinematics for a target pose.
    pub fn ik(&self, pose: &Pose) -> Result<IkResult> {
        self.handle.ik(pose)
    }

    /// Compute forward kinematics for a joint configuration.
    pub fn fk(&self, q: &[f64]) -> Result<Pose> {
        self.handle.fk(q)
    }

    /// Whether the robot has a spherical wrist.
    pub fn is_spherical(&self) -> bool {
        self.handle.is_spherical()
    }

    /// Whether EAIK can solve IK for this robot analytically.
    pub fn has_known_decomposition(&self) -> bool {
        self.handle.has_known_decomposition()
    }

    /// Descriptive string of the robot's kinematic family.
    pub fn kinematic_family(&self) -> String {
        self.handle.kinematic_family()
    }

    /// Number of joints.
    pub fn n_joints(&self) -> usize {
        self.handle.n_joints as usize
    }

    /// Joint axes after kinematic remodeling (3 x n, column-major).
    pub fn remodeled_h(&self) -> Matrix {
        self.handle.remodeled_h()
    }

    /// Joint offsets after kinematic remodeling (3 x (n+1), column-major).
    pub fn remodeled_p(&self) -> Matrix {
        self.handle.remodeled_p()
    }

    /// Original joint axes (3 x n, column-major).
    pub fn original_h(&self) -> Matrix {
        self.handle.original_h()
    }

    /// Original joint offsets (3 x (n+1), column-major).
    pub fn original_p(&self) -> Matrix {
        self.handle.original_p()
    }
}

// ---------------------------------------------------------------------------
// DhRobot
// ---------------------------------------------------------------------------

/// A robot defined by standard Denavit-Hartenberg parameters.
///
/// Each parameter array has length equal to the number of joints.
pub struct DhRobot {
    handle: RobotHandle,
}

impl DhRobot {
    /// Create a new DH robot.
    ///
    /// - `alpha`: DH alpha parameters (twist angles, radians).
    /// - `a`: DH a parameters (link lengths).
    /// - `d`: DH d parameters (link offsets).
    ///
    /// All slices must have the same length.
    pub fn new(alpha: &[f64], a: &[f64], d: &[f64]) -> Result<Self> {
        Self::new_with_options(alpha, a, d, None, &[], true)
    }

    /// Create a DH robot with full options.
    pub fn new_with_options(
        alpha: &[f64],
        a: &[f64],
        d: &[f64],
        r6t: Option<&[f64; 9]>,
        fixed_axes: &[FixedAxis],
        double_precision: bool,
    ) -> Result<Self> {
        assert_eq!(alpha.len(), a.len(), "alpha and a must have the same length");
        assert_eq!(alpha.len(), d.len(), "alpha and d must have the same length");
        let n_joints = alpha.len() as u32;

        let (fi, fv) = parse_fixed_axes(fixed_axes);
        let (fi_ptr, fv_ptr, nf) = fixed_ptrs(&fi, &fv);
        let r6t_ptr = r6t.map_or(std::ptr::null(), |r| r.as_ptr());

        let mut err = [0u8; 512];
        let ptr = unsafe {
            ffi::eaik_robot_new_dh(
                alpha.as_ptr(),
                a.as_ptr(),
                d.as_ptr(),
                n_joints,
                r6t_ptr,
                fi_ptr,
                fv_ptr,
                nf,
                double_precision,
                err.as_mut_ptr().cast(),
                512,
            )
        };
        if ptr.is_null() {
            return Err(Error(read_err_buf(&err)));
        }
        Ok(Self {
            handle: RobotHandle { ptr, n_joints },
        })
    }

    /// Compute inverse kinematics for a target pose.
    pub fn ik(&self, pose: &Pose) -> Result<IkResult> {
        self.handle.ik(pose)
    }

    /// Compute forward kinematics for a joint configuration.
    pub fn fk(&self, q: &[f64]) -> Result<Pose> {
        self.handle.fk(q)
    }

    /// Whether the robot has a spherical wrist.
    pub fn is_spherical(&self) -> bool {
        self.handle.is_spherical()
    }

    /// Whether EAIK can solve IK for this robot analytically.
    pub fn has_known_decomposition(&self) -> bool {
        self.handle.has_known_decomposition()
    }

    /// Descriptive string of the robot's kinematic family.
    pub fn kinematic_family(&self) -> String {
        self.handle.kinematic_family()
    }

    /// Number of joints.
    pub fn n_joints(&self) -> usize {
        self.handle.n_joints as usize
    }

    /// Joint axes after kinematic remodeling.
    pub fn remodeled_h(&self) -> Matrix {
        self.handle.remodeled_h()
    }

    /// Joint offsets after kinematic remodeling.
    pub fn remodeled_p(&self) -> Matrix {
        self.handle.remodeled_p()
    }

    /// Original joint axes.
    pub fn original_h(&self) -> Matrix {
        self.handle.original_h()
    }

    /// Original joint offsets.
    pub fn original_p(&self) -> Matrix {
        self.handle.original_p()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn puma_dh() -> DhRobot {
        let pi = std::f64::consts::PI;
        // Puma parameters from examples/dh_ik_example.py
        let alpha = [-pi / 2.0, 0.0, pi / 2.0, -pi / 2.0, pi / 2.0, 0.0];
        let a = [0.0, 0.43180, -0.02032, 0.0, 0.0, 0.0];
        let d = [0.67183, 0.13970, 0.0, 0.43180, 0.0, 0.0565];
        DhRobot::new(&alpha, &a, &d).expect("Failed to create Puma")
    }

    #[test]
    fn test_dh_construction() {
        let robot = puma_dh();
        assert!(robot.has_known_decomposition());
        assert_eq!(robot.n_joints(), 6);
    }

    #[test]
    fn test_fk_pose_valid() {
        let robot = puma_dh();
        let q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let pose = robot.fk(&q).unwrap();

        // Bottom row of homogeneous matrix should be [0, 0, 0, 1]
        assert!((pose.get(3, 0)).abs() < 1e-10);
        assert!((pose.get(3, 1)).abs() < 1e-10);
        assert!((pose.get(3, 2)).abs() < 1e-10);
        assert!((pose.get(3, 3) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ik_roundtrip() {
        let robot = puma_dh();
        let q_original = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let pose = robot.fk(&q_original).unwrap();
        let result = robot.ik(&pose).unwrap();

        assert!(!result.is_empty(), "Should find at least one IK solution");

        // At least one solution should reconstruct the original FK
        let mut found_match = false;
        for sol in &result {
            let recon = robot.fk(&sol.joints).unwrap();
            let mut max_err = 0.0f64;
            for i in 0..16 {
                max_err = max_err.max((recon.data[i] - pose.data[i]).abs());
            }
            if max_err < 1e-6 {
                found_match = true;
                break;
            }
        }
        assert!(found_match, "No IK solution matches the original FK pose");
    }

    #[test]
    fn test_hp_construction() {
        // 2R planar robot: two Z-axis joints with unit offsets along X
        // H: 3x2 column-major: col0 = [0,0,1], col1 = [0,0,1]
        let h = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        // P: 3x3 column-major: p0=[0,0,0], p1=[1,0,0], p2=[1,0,0]
        let p = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

        let robot = HpRobot::new(&h, &p).unwrap();
        assert!(robot.has_known_decomposition());
        assert_eq!(robot.n_joints(), 2);
    }

    #[test]
    fn test_kinematic_family() {
        let robot = puma_dh();
        let family = robot.kinematic_family();
        assert!(!family.is_empty());
    }

    #[test]
    fn test_matrix_accessors() {
        let robot = puma_dh();
        let h = robot.original_h();
        assert_eq!(h.rows(), 3);
        assert_eq!(h.cols(), 6);

        let p = robot.original_p();
        assert_eq!(p.rows(), 3);
        assert_eq!(p.cols(), 7);
    }

    #[test]
    fn test_exact_solutions_filter() {
        let robot = puma_dh();
        let q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let pose = robot.fk(&q).unwrap();
        let result = robot.ik(&pose).unwrap();
        let exact = result.exact_solutions();
        // All exact solutions should have is_least_squares == false
        for sol in &exact {
            assert!(!sol.is_least_squares);
        }
    }

    #[test]
    fn test_is_spherical() {
        let robot = puma_dh();
        // Puma 560 has a spherical wrist
        assert!(robot.is_spherical());
    }
}
