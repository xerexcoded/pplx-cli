from pplx_cli.local import local_only_enabled, local_only_mode


def test_nested_scope_cannot_relax_local_policy():
    assert not local_only_enabled()
    with local_only_mode():
        with local_only_mode(False):
            assert local_only_enabled()
        assert local_only_enabled()
    assert not local_only_enabled()


def test_policy_resets_after_failure():
    try:
        with local_only_mode():
            raise RuntimeError("failed operation")
    except RuntimeError:
        pass
    assert not local_only_enabled()
