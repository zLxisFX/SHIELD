def test_shield_import():
    import shield
    assert isinstance(shield.__version__, str)
