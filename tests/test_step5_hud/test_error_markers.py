def test_error_markers_import():
    from barpath.pipeline.step5_helpers.error_markers import draw_error_markers

    assert callable(draw_error_markers)
