#ifndef EAIK_FFI_H
#define EAIK_FFI_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EaikRobot EaikRobot;
typedef struct EaikIKSolution EaikIKSolution;
typedef struct EaikMatrix EaikMatrix;

// --- Robot construction ---

// Create robot from H (3 x n_joints, col-major) and P (3 x (n_joints+1), col-major).
// r6t: 3x3 col-major rotation matrix (or NULL for identity).
// fixed_axes_indices/values: arrays of length n_fixed.
// Returns NULL on error, writing message to err_buf.
EaikRobot *eaik_robot_new_hp(
    const double *h_data, uint32_t n_joints,
    const double *p_data,
    const double *r6t, // NULL = identity
    const int32_t *fixed_axes_indices, const double *fixed_axes_values, uint32_t n_fixed,
    bool is_double_precision,
    char *err_buf, uint32_t err_buf_len);

// Create robot from DH parameters (arrays of length n_joints).
EaikRobot *eaik_robot_new_dh(
    const double *dh_alpha, const double *dh_a, const double *dh_d, uint32_t n_joints,
    const double *r6t, // NULL = identity
    const int32_t *fixed_axes_indices, const double *fixed_axes_values, uint32_t n_fixed,
    bool is_double_precision,
    char *err_buf, uint32_t err_buf_len);

void eaik_robot_free(EaikRobot *robot);

// --- Inverse Kinematics ---

// pose: 4x4 col-major. Returns NULL on error.
EaikIKSolution *eaik_robot_ik(const EaikRobot *robot, const double *pose,
                               char *err_buf, uint32_t err_buf_len);

void eaik_ik_solution_free(EaikIKSolution *sol);
uint32_t eaik_ik_solution_num_solutions(const EaikIKSolution *sol);
uint32_t eaik_ik_solution_num_joints(const EaikIKSolution *sol);
// Copy joint angles for solution at index idx into out_q (length = num_joints).
void eaik_ik_solution_get_q(const EaikIKSolution *sol, uint32_t idx, double *out_q);
bool eaik_ik_solution_is_ls(const EaikIKSolution *sol, uint32_t idx);

// --- Forward Kinematics ---

// q: array of n_joints joint angles. out_pose: 4x4 col-major output.
int32_t eaik_robot_fwdkin(const EaikRobot *robot, const double *q, uint32_t n_joints,
                           double *out_pose, char *err_buf, uint32_t err_buf_len);

// --- Robot queries ---

bool eaik_robot_is_spherical(const EaikRobot *robot);
bool eaik_robot_has_known_decomposition(const EaikRobot *robot);

// Returns length written (excluding null), or -1 on error. Always null-terminates if buf_len > 0.
int32_t eaik_robot_get_kinematic_family(const EaikRobot *robot, char *buf, uint32_t buf_len);

// --- Matrix accessors ---
// Returns a heap-allocated EaikMatrix. Caller must free with eaik_matrix_free.
EaikMatrix *eaik_robot_get_remodeled_h(const EaikRobot *robot);
EaikMatrix *eaik_robot_get_remodeled_p(const EaikRobot *robot);
EaikMatrix *eaik_robot_get_original_h(const EaikRobot *robot);
EaikMatrix *eaik_robot_get_original_p(const EaikRobot *robot);

void eaik_matrix_free(EaikMatrix *mat);
uint32_t eaik_matrix_rows(const EaikMatrix *mat);
uint32_t eaik_matrix_cols(const EaikMatrix *mat);
// Returns pointer to col-major data (rows * cols doubles). Valid until eaik_matrix_free.
const double *eaik_matrix_data(const EaikMatrix *mat);

#ifdef __cplusplus
}
#endif

#endif // EAIK_FFI_H
