#!/usr/bin/env python3
# Author: Daniel Ostermeier
# Date: 09.11.23

from abc import ABC

import numpy as np

from urchin import URDF

import eaik.pybindings.EAIK as native

__all__ = ["IKRobot", "UrdfRobot", "HPRobot", "HomogeneousRobot", "DhRobot", "native"]


class IKRobot(ABC):
    """An interface for the python-side robot implementation of this toolbox"""

    _robot: native.Robot

    @staticmethod
    def urdf_to_sp_conv(axis_trafo, axis, parent_p):
        """
        Convert urchin axis to axis-translation convention for subproblems

        :param axis_trafo: 4x4 homogeneous transformation of a joint w.r.t. a world frame
        :param axis: Joint axis within axis_trafo (e.g., z-axis)
        :param parent_p: Linear global offset of last joint
        :return: (Axis vector, translation)
        """
        R = axis_trafo[:-1, :-1]  # Rotation in global basis frame
        T = axis_trafo[:-1, -1] - parent_p  # Translation in local joint-frame
        axis_n = R.dot(axis)
        return axis_n, T

    def hasSphericalWrist(self) -> bool:
        """Is true iff the robot has a spherical wrist."""
        return self._robot.is_spherical()

    def hasKnownDecomposition(self) -> bool:
        """Is true iff there is a geometric decomposition that allows solving IK for this robot."""
        return self._robot.has_known_decomposition()

    def getKinematicFamily(self) -> str:
        """Returns information on the kinematic family of the robot."""
        return self._robot.get_kinematic_family()

    def getRemodeled_H(self) -> np.ndarray:
        """Returns the joint axes of this robot after kinematic remodeling"""
        return self._robot.get_remodeled_H()

    def getRemodeled_P(self) -> np.ndarray:
        """Returns the joint axes' reference points for this robot after kinematic remodeling"""
        return self._robot.get_remodeled_P()

    def getOriginal_H(self) -> np.ndarray:
        """Returns the original joint axes of this robot before kinematic remodeling"""
        return self._robot.get_original_H()

    def getOriginal_P(self) -> np.ndarray:
        """Returns the original joint axes' reference points for this robot before kinematic remodeling"""
        return self._robot.get_original_P()

    def fwdKin(self, q: np.ndarray) -> np.ndarray:
        """Calculate the forward kinematics for a given joint configuration"""
        return self._robot.fwdkin(q)

    def IK(self, pose: np.ndarray):
        """
        Calculate the inverse kinematics for a desired pose

        :param pose:  4x4 Homogeneous transformation matrix
        """
        return self._robot.calculate_IK(pose)

    def IK_batched(self, pose_batch, num_worker_threads=4):
        """
        Returns multiple IK for a batch of desired poses.

        :param pose_batch: A numpy array with shape (batch_size, 4, 4) or a list of 2D numpy arrays with shape (4, 4)
        :param num_worker_threads: Number of worker threads to use for parallel IK calculation
        """
        # Check if the input is a pure numpy array
        if isinstance(pose_batch, np.ndarray) and pose_batch.ndim == 3 and pose_batch.shape[1:] == (4, 4):
            pose_list = [pose_batch[i] for i in range(pose_batch.shape[0])]
        elif isinstance(pose_batch, list) and all(isinstance(p, np.ndarray) and p.shape == (4, 4) for p in pose_batch):
            pose_list = pose_batch
        else:
            raise ValueError("Input must be a numpy array with shape (batch_size, 4, 4) or a list of 2D numpy arrays "
                             "with shape (4, 4)")
        return self._robot.calculate_IK_batched(pose_list, num_worker_threads)

class UrdfRobot(IKRobot):
    """A robot for which the kinematic chain is parsed from a URDF file."""

    def __init__(self,
                 file_path: str,
                 fixed_axes: list[tuple[int, float]] | None = None):
        """
        EAIK Robot parametrized by URDF file

        :param file_path: Path to URDF file
        :param fixed_axes: List of tuples defining fixed joints (zero-indexed) (i, q_i+1)
        """
        if fixed_axes is None:
            fixed_axes = []
        super().__init__()
        robot = URDF.load(file_path, lazy_load_meshes=True)
        joints = robot._sort_joints(robot.actuated_joints)

        fk_zero_pose = robot.link_fk()  # Calculate FK

        parent_p = np.zeros(3)
        H = np.array([], dtype=np.int64).reshape(0, 3)  # axes
        P = np.array([], dtype=np.int64).reshape(0, 3)  # offsets
        for i in range(len(joints)):
            joint_child_link = robot.link_map[joints[i].child]
            h, p = self.urdf_to_sp_conv(fk_zero_pose[joint_child_link], joints[i].axis, parent_p)
            H = np.vstack([H, h])
            P = np.vstack([P, p])
            parent_p += p

        # End effector displacement is (0,0,0)
        P = np.vstack([P, np.zeros(3)])
        self._robot = native.Robot(H.T, P.T, np.eye(3), fixed_axes, True)


class HPRobot(IKRobot):
    """A robot parameterized by by H and P vectors"""

    def __init__(self,
                 H: np.ndarray,
                 P: np.ndarray,
                 fixed_axes: list[tuple[int, float]] | None = None):
        """
        EAIK Robot parametrized by H and P vectors

        :param H: (N, 3) numpy array resembling the unit direction vectors of the joints
        :param P: (N+1, 3) numpy array resembling the offsets between the joint axes 
        :param fixed_axes: List of tuples defining fixed joints (zero-indexed) (i, q_i+1)
        """
        super().__init__()
        if fixed_axes is None:
            fixed_axes = []

        self._robot = native.Robot(H.T, P.T, np.eye(3), fixed_axes, True)


class HomogeneousRobot(IKRobot):
    """A robot parametrized by homogeneous joint transformations"""

    def __init__(self,
                 joint_trafos: np.ndarray,
                 fixed_axes: list[tuple[int, float]] | None = None,
                 joint_axis: np.ndarray | None = None):
        """
        EAIK Robot parametrized by homogeneous joint transformations

        :param joint_trafos: (N+1)x4x4 numpy array of homogeneous transformations of each of N joints w.r.t. the world
            frame, i.e. (T01, T02, ..., T0EE)
        :param fixed_axes: List of tuples defining fixed joints (zero-indexed) (i, q_i+1)
        :param joint_axis: Unit vector of joint axis orientation within each frame in joint_trafos (e.g., z-axis)
        """
        super().__init__()
        if fixed_axes is None:
            fixed_axes = []
        if joint_axis is None:
            joint_axis = np.array([0, 0, 1], dtype=np.float64)

        H = np.array([], dtype=np.float64).reshape(0, 3)  # axes
        P = np.array([], dtype=np.float64).reshape(0, 3)  # offsets

        # Derive Kinematic from zero-position
        parent_p = np.zeros(3, dtype=np.float64)
        for frame in joint_trafos[:-1]:
            h, p = IKRobot.urdf_to_sp_conv(frame, joint_axis, parent_p)
            H = np.vstack([H, h])
            P = np.vstack([P, p])
            parent_p += p
            
        # Account for end effector pose
        p_EE = joint_trafos[-1][:-1, -1] - joint_trafos[-2][:-1, -1] # Translation in local joint-frame
        P = np.vstack([P, p_EE])
        rNt = joint_trafos[-1][:-1, :-1]  # Rotation in global basis frame

        self._robot = native.Robot(H.T, P.T, rNt, fixed_axes, True)


class DhRobot(IKRobot):
    """A robot parameterized by standard Denavit-Hartenberg parameters"""

    def __init__(self,
                 dh_alpha: np.ndarray,
                 dh_a: np.ndarray,
                 dh_d: np.ndarray,
                 fixed_axes: list[tuple[int, float]] | None = None):
        """
        EAIK Robot parametrized by homogeneous joint transformations

        :param dh_alpha
        :param dh_a
        :param dh_d
        :param fixed_axes: List of tuples defining fixed joints (zero-indexed) (i, q_i+1)
        """
        super().__init__()
        if fixed_axes is None:
            fixed_axes = []

        self._robot = native.Robot(dh_alpha, dh_a, dh_d, np.eye(3), fixed_axes, True)