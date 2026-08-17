def test_power_band_import():
    from barpath.pipeline.step5_helpers.power_band import draw_power_zone_band

    assert callable(draw_power_zone_band)
