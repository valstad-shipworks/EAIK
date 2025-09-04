from pathlib import Path
import os
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Optional: let users point to Eigen with an env var
eigen_hint = os.environ.get("EIGEN3_INCLUDE_DIR")
include_dirs = [
    "CPP/external/ik-geo/cpp/subproblems",
    "CPP/src/IK",
    "CPP/src/IK/utils",
    "CPP/src",
    "CPP/src/utils",
]
if eigen_hint:
    include_dirs.append(eigen_hint)
else:
    # Fallbacks (only if they exist)
    for p in ("/usr/include/eigen3", "/usr/local/include/eigen3"):
        if Path(p).exists():
            include_dirs.append(p)

sources = [
    "src/eaik/pybindings/eaik_pybindings.cpp",
    "CPP/src/IK/utils/kinematic_utils.cpp",
    "CPP/src/IK/1R_IK.cpp",
    "CPP/src/IK/2R_IK.cpp",
    "CPP/src/IK/3R_IK.cpp",
    "CPP/src/IK/4R_IK.cpp",
    "CPP/src/IK/5R_IK.cpp",
    "CPP/src/IK/6R_IK.cpp",
    "CPP/src/EAIK.cpp",
    "CPP/src/utils/kinematic_remodeling.cpp",
    "CPP/external/ik-geo/cpp/subproblems/sp.cpp",
]

ext_modules = [
    Pybind11Extension(
        "eaik.pybindings.EAIK",
        sorted(sources),
        include_dirs=include_dirs,
        cxx_std=17,                 # <-- portable C++ standard
        # define_macros=[("NOMINMAX", 1)]  # uncomment on Windows if needed
        # extra_link_args=[...]            # add platform-specific linker flags if needed
    )
]

setup(
    name="eaik",
    version="0.1.0",
    description="Efficient Analytical IK bindings",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["eaik*"]),  # ensures eaik & eaik.pybindings get installed
    py_modules=["eaik"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
