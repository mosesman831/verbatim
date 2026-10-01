"""SPEC_V4_5 acceptance-scenario tests (§15, D01–D24).

Public-surface tests against real ``Store.create`` fixtures — the same
harness shape as ``tests/v4/test_scenarios_*.py``. Implemented v4.5
mechanisms are exercised through their production entry points; deferred
or absent mechanisms assert the honest ``CAPABILITY_UNAVAILABLE`` /
absent-surface contract instead of skipping.
"""
