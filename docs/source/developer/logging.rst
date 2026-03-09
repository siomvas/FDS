Logging Policy
==============

FDS follows a layered logging contract to keep runtime behavior predictable:

1. Library modules

- Use module-level loggers: ``log = logging.getLogger(__name__)``.
- Do not use ``print`` for operational messages.
- Emit actionable exception messages when re-raising failures.

2. CLI/script entrypoints

- Configure logging once in ``main()`` with ``logging.basicConfig(...)``.
- Use ``INFO`` for progress, ``WARNING`` for recoverable degradation, ``ERROR`` for failed units.
- Support ``--dry-run`` for side-effect free validation where applicable.

3. Warning semantics

- Use ``warnings.warn(..., RuntimeWarning)`` for non-fatal sample/data quality issues.
- Raise typed exceptions for hard failures.

4. Exception handling

- Avoid ``except Exception`` in runtime paths unless a boundary must protect orchestration.
- Prefer explicit exception tuples and include file/config context in log output.

5. Tests

- Warning paths should be asserted with ``pytest.warns``.
- New CLI behaviors should include argument contract tests for canonical flags.
