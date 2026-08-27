# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Missing from the upstream public release entirely (not just an empty stub --
# the module doesn't exist at all). Only referenced from gen_plan.py's
# subtask == 'parse_result' branch (re-analyzing a PAST run's saved output),
# which the Hybrid-NL2SVA UART pilot never exercises (we use subtask ==
# 'actual_gen' and swap in our own NL2SVA pipeline afterward) -- these stubs
# exist solely so the unconditional module-level import in gen_plan.py doesn't
# crash; none of them are meant to be called for our path.


def _unimplemented(name):
    raise NotImplementedError(
        f"{name} was never open-sourced upstream and is unused by the "
        "'actual_gen' subtask this pilot exercises -- only wire it up if you "
        "actually need 'parse_result'."
    )


def load_pdf_stats(load_dir):
    _unimplemented("load_pdf_stats")


def load_nl_plans(load_dir):
    _unimplemented("load_nl_plans")


def load_svas(load_dir):
    _unimplemented("load_svas")


def load_jasper_reports(load_dir):
    _unimplemented("load_jasper_reports")
