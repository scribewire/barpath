def test_joint_angles_import():
    from barpath.pipeline.step5_helpers.joint_angles import draw_knee_angles
    assert callable(draw_knee_angles)


def test_angle_color_green():
    from barpath.pipeline.step5_helpers.joint_angles import _get_angle_color
    from barpath.pipeline.config import ANGLE_GREEN_BGR
    assert _get_angle_color(110, 90, 135) == ANGLE_GREEN_BGR
