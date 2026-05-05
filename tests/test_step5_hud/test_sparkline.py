def test_sparkline_import():
    from barpath.pipeline.step5_helpers.sparkline import draw_velocity_sparkline
    assert callable(draw_velocity_sparkline)
