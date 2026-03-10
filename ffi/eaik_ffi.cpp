#include "eaik_ffi.h"
#include "EAIK.h"

#include <cstring>
#include <string>
#include <vector>

// --- Opaque types ---

struct EaikRobot {
    EAIK::Robot inner;

    // HP constructor
    EaikRobot(const Eigen::MatrixXd &H, const Eigen::MatrixXd &P,
              const Eigen::Matrix3d &R6T,
              const std::vector<std::pair<int, double>> &fixed,
              bool double_prec)
        : inner(H, P, R6T, fixed, double_prec) {}

    // DH constructor
    EaikRobot(const Eigen::VectorXd &alpha, const Eigen::VectorXd &a,
              const Eigen::VectorXd &d, const Eigen::Matrix3d &R6T,
              const std::vector<std::pair<int, double>> &fixed,
              bool double_prec)
        : inner(alpha, a, d, R6T, fixed, double_prec) {}
};

struct EaikIKSolution {
    IKS::IK_Solution sol;
};

struct EaikMatrix {
    Eigen::MatrixXd mat;
};

// --- Helpers ---

static void write_err(const char *msg, char *buf, uint32_t len) {
    if (buf && len > 0) {
        std::strncpy(buf, msg, len - 1);
        buf[len - 1] = '\0';
    }
}

static Eigen::Matrix3d parse_r6t(const double *r6t) {
    if (r6t) {
        return Eigen::Map<const Eigen::Matrix<double, 3, 3>>(r6t);
    }
    return Eigen::Matrix3d::Identity();
}

static std::vector<std::pair<int, double>> parse_fixed(
    const int32_t *indices, const double *values, uint32_t n) {
    std::vector<std::pair<int, double>> fixed;
    fixed.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        fixed.emplace_back(static_cast<int>(indices[i]), values[i]);
    }
    return fixed;
}

// --- Robot construction ---

extern "C" EaikRobot *eaik_robot_new_hp(
    const double *h_data, uint32_t n_joints,
    const double *p_data,
    const double *r6t,
    const int32_t *fixed_axes_indices, const double *fixed_axes_values, uint32_t n_fixed,
    bool is_double_precision,
    char *err_buf, uint32_t err_buf_len) {
    try {
        // H is 3 x n_joints, col-major
        Eigen::Map<const Eigen::MatrixXd> H(h_data, 3, n_joints);
        // P is 3 x (n_joints + 1), col-major
        Eigen::Map<const Eigen::MatrixXd> P(p_data, 3, n_joints + 1);
        auto R6T = parse_r6t(r6t);
        auto fixed = parse_fixed(fixed_axes_indices, fixed_axes_values, n_fixed);
        return new EaikRobot(H, P, R6T, fixed, is_double_precision);
    } catch (const std::exception &e) {
        write_err(e.what(), err_buf, err_buf_len);
        return nullptr;
    }
}

extern "C" EaikRobot *eaik_robot_new_dh(
    const double *dh_alpha, const double *dh_a, const double *dh_d, uint32_t n_joints,
    const double *r6t,
    const int32_t *fixed_axes_indices, const double *fixed_axes_values, uint32_t n_fixed,
    bool is_double_precision,
    char *err_buf, uint32_t err_buf_len) {
    try {
        Eigen::Map<const Eigen::VectorXd> alpha(dh_alpha, n_joints);
        Eigen::Map<const Eigen::VectorXd> a(dh_a, n_joints);
        Eigen::Map<const Eigen::VectorXd> d(dh_d, n_joints);
        auto R6T = parse_r6t(r6t);
        auto fixed = parse_fixed(fixed_axes_indices, fixed_axes_values, n_fixed);
        return new EaikRobot(alpha, a, d, R6T, fixed, is_double_precision);
    } catch (const std::exception &e) {
        write_err(e.what(), err_buf, err_buf_len);
        return nullptr;
    }
}

extern "C" void eaik_robot_free(EaikRobot *robot) {
    delete robot;
}

// --- Inverse Kinematics ---

extern "C" EaikIKSolution *eaik_robot_ik(
    const EaikRobot *robot, const double *pose,
    char *err_buf, uint32_t err_buf_len) {
    try {
        Eigen::Map<const IKS::Homogeneous_T> T(pose);
        auto *sol = new EaikIKSolution();
        sol->sol = robot->inner.calculate_IK(T);
        return sol;
    } catch (const std::exception &e) {
        write_err(e.what(), err_buf, err_buf_len);
        return nullptr;
    }
}

extern "C" void eaik_ik_solution_free(EaikIKSolution *sol) {
    delete sol;
}

extern "C" uint32_t eaik_ik_solution_num_solutions(const EaikIKSolution *sol) {
    return static_cast<uint32_t>(sol->sol.Q.size());
}

extern "C" uint32_t eaik_ik_solution_num_joints(const EaikIKSolution *sol) {
    if (sol->sol.Q.empty()) return 0;
    return static_cast<uint32_t>(sol->sol.Q[0].size());
}

extern "C" void eaik_ik_solution_get_q(const EaikIKSolution *sol, uint32_t idx, double *out_q) {
    const auto &q = sol->sol.Q[idx];
    std::memcpy(out_q, q.data(), q.size() * sizeof(double));
}

extern "C" bool eaik_ik_solution_is_ls(const EaikIKSolution *sol, uint32_t idx) {
    return sol->sol.is_LS_vec[idx];
}

// --- Forward Kinematics ---

extern "C" int32_t eaik_robot_fwdkin(
    const EaikRobot *robot, const double *q, uint32_t n_joints,
    double *out_pose, char *err_buf, uint32_t err_buf_len) {
    try {
        std::vector<double> Q(q, q + n_joints);
        IKS::Homogeneous_T T = robot->inner.fwdkin(Q);
        // Copy col-major 4x4
        Eigen::Map<Eigen::Matrix4d> out_map(out_pose);
        out_map = T;
        return 0;
    } catch (const std::exception &e) {
        write_err(e.what(), err_buf, err_buf_len);
        return -1;
    }
}

// --- Robot queries ---

extern "C" bool eaik_robot_is_spherical(const EaikRobot *robot) {
    return robot->inner.is_spherical();
}

extern "C" bool eaik_robot_has_known_decomposition(const EaikRobot *robot) {
    return robot->inner.has_known_decomposition();
}

extern "C" int32_t eaik_robot_get_kinematic_family(const EaikRobot *robot, char *buf, uint32_t buf_len) {
    std::string family = robot->inner.get_kinematic_family();
    if (buf_len == 0) return -1;
    uint32_t copy_len = static_cast<uint32_t>(family.size());
    if (copy_len >= buf_len) copy_len = buf_len - 1;
    std::memcpy(buf, family.c_str(), copy_len);
    buf[copy_len] = '\0';
    return static_cast<int32_t>(copy_len);
}

// --- Matrix accessors ---

extern "C" EaikMatrix *eaik_robot_get_remodeled_h(const EaikRobot *robot) {
    auto *m = new EaikMatrix();
    m->mat = robot->inner.get_remodeled_H();
    return m;
}

extern "C" EaikMatrix *eaik_robot_get_remodeled_p(const EaikRobot *robot) {
    auto *m = new EaikMatrix();
    m->mat = robot->inner.get_remodeled_P();
    return m;
}

extern "C" EaikMatrix *eaik_robot_get_original_h(const EaikRobot *robot) {
    auto *m = new EaikMatrix();
    m->mat = robot->inner.get_original_H();
    return m;
}

extern "C" EaikMatrix *eaik_robot_get_original_p(const EaikRobot *robot) {
    auto *m = new EaikMatrix();
    m->mat = robot->inner.get_original_P();
    return m;
}

extern "C" void eaik_matrix_free(EaikMatrix *mat) {
    delete mat;
}

extern "C" uint32_t eaik_matrix_rows(const EaikMatrix *mat) {
    return static_cast<uint32_t>(mat->mat.rows());
}

extern "C" uint32_t eaik_matrix_cols(const EaikMatrix *mat) {
    return static_cast<uint32_t>(mat->mat.cols());
}

extern "C" const double *eaik_matrix_data(const EaikMatrix *mat) {
    return mat->mat.data();
}
