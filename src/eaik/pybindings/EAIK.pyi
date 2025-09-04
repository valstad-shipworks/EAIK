"""

    Pybind11 of EAIK
    
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['IKSolution', 'Robot']
class IKSolution:
    def __bool__(self) -> bool:
        ...
    def __init__(self) -> None:
        ...
    def num_solutions(self) -> int:
        """
                    Returns number of total IK solutions (including Least-Squares)
                    :return:   int
        """
    @property
    def Q(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        ...
    @Q.setter
    def Q(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"]) -> None:
        ...
    @property
    def is_LS(self) -> typing.Annotated[numpy.typing.NDArray[numpy.bool], "[m, 1]"]:
        ...
    @is_LS.setter
    def is_LS(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.bool, "[m, 1]"]) -> None:
        ...
class Robot:
    @typing.overload
    def __init__(self, H: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], P: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], R6T: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 3]"], fixed_axes: collections.abc.Sequence[tuple[typing.SupportsInt, typing.SupportsFloat]], use_double_precision: bool) -> None:
        """
                    The EAIK Robot class.
        
                    :param H:  Unit vectors defining the joint axes
                    :param P:  Linear Joint offsets
                    :param R6T:  Endeffector orientation w.r.t. joint 6
                    :param fixed_axes:  List of tuples defining fixed joints (zero-indexed) (i, q_i+1)    
                    :param use_double_precision:  Use double precision (standard)
        """
    @typing.overload
    def __init__(self, dh_alpha: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], dh_a: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], dh_d: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], R6T: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 3]"], fixed_axes: collections.abc.Sequence[tuple[typing.SupportsInt, typing.SupportsFloat]], use_double_precision: bool) -> None:
        """
                    The EAIK Robot class.
        
                    :param dh_alpha:  DH-Parameters: alpha
                    :param dh_a:  DH-Parameters: a
                    :param dh_d:  DH-Parameters: d
                    :param R6T:  Endeffector orientation w.r.t. joint 6
                    :param fixed_axes:  List of tuples defining fixed joints (zero-indexed) (i, q_i+1)    
                    :param use_double_precision:  Use double precision (standard)
        """
    def calculate_IK(self, pose: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[4, 4]"]) -> IKSolution:
        """
                    Run inverse kinematics.
        
                    :param pose:  4x4 Homogeneous transformation matrix
                    :return:      IK-Solution class
        """
    def calculate_IK_batched(self, pose_batch: collections.abc.Sequence[typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[4, 4]"]], num_worker_threads: typing.SupportsInt) -> list[IKSolution]:
        """
                    Run inverse kinematics for a batch of EE poses.
        
                    :param pose:  Batch of 4x4 Homogeneous transformation matrix
                    :param num_worker_threads: Number of total worker threads to assign
                    :return:      Batch of IK-Solution classes
        """
    def fwdkin(self, Q: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 4]"]:
        """
                    Run forward kinematics.
        
                    :param Q:  Array of six joint angles
                    :return:   4x4 Homogeneous transformation matrix
        """
    def get_kinematic_family(self) -> str:
        ...
    def get_original_H(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        ...
    def get_original_P(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        ...
    def get_remodeled_H(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        ...
    def get_remodeled_P(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        ...
    def has_known_decomposition(self) -> bool:
        """
                    Returns if robot has a known SP decomposition
                    :return:   bool
        """
    def is_spherical(self) -> bool:
        """
                    Returns if robot has spherical wrist.
                    :return:   bool
        """
