from load_test import percentile


def test_percentile_xu_ly_danh_sach_rong_va_gia_tri_bien():
    assert percentile([], 0.95) == 0.0
    assert percentile([10.0, 20.0, 30.0, 40.0], 0.50) == 20.0
    assert percentile([10.0, 20.0, 30.0, 40.0], 0.95) == 40.0
