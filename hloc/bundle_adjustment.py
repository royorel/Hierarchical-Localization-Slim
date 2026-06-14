"""Bundle adjustment over an existing COLMAP reconstruction (pycolmap backend).

Extrinsics-only BA with the similarity gauge -- rotation, translation, AND scale --
pinned via BundleAdjustmentGauge.THREE_POINTS, so the reconstruction stays in its
input world frame. This is the right primitive for refining slightly-inconsistent
provided poses (e.g. handheld multi-view rigs) without drifting scale, which would
otherwise desync the refined poses from the rest of the pipeline.
"""
from pathlib import Path
from typing import Optional, Union

import pycolmap


def main(
    model: Union[pycolmap.Reconstruction, str, Path],
    output: Optional[Union[str, Path]] = None,
    refine_extrinsics: bool = True,
    refine_intrinsics: bool = False,
    fix_gauge: bool = True,
) -> pycolmap.Reconstruction:
    """Bundle-adjust `model` (a Reconstruction or a path to one) and return it.

    refine_extrinsics : optimize the per-image rig_from_world poses.
    refine_intrinsics : optimize focal length, principal point, extra params.
    fix_gauge         : pin the 7-DOF similarity gauge (incl scale) via THREE_POINTS,
                        keeping the result in the input world frame.
    output            : if given, write the adjusted reconstruction there.
    """
    rec = (model if isinstance(model, pycolmap.Reconstruction)
           else pycolmap.Reconstruction(str(model)))

    config = pycolmap.BundleAdjustmentConfig()
    for image_id in rec.reg_image_ids():
        config.add_image(image_id)
    if fix_gauge:
        config.fix_gauge(pycolmap.BundleAdjustmentGauge.THREE_POINTS)

    options = pycolmap.BundleAdjustmentOptions()
    options.refine_rig_from_world = refine_extrinsics
    options.refine_sensor_from_rig = False
    options.refine_focal_length = refine_intrinsics
    options.refine_principal_point = refine_intrinsics
    options.refine_extra_params = refine_intrinsics
    options.print_summary = False

    pycolmap.create_default_bundle_adjuster(options, config, rec).solve()

    if output is not None:
        Path(output).mkdir(parents=True, exist_ok=True)
        rec.write(str(output))
    return rec
