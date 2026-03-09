Testing
=======

Quick-pass checks:

.. code-block:: bash

   find . -type f -name '*.py' -not -path '*/__pycache__/*' -print0 | xargs -0 python3 -m py_compile
   python3 -m pytest --collect-only -q

Targeted suites:

.. code-block:: bash

   python3 -m pytest tests/test_data_modules.py -q
   python3 -m pytest tests/test_detection_data_module.py -q

Notes:

- Some suites require optional dependency extras (`embedding`, `evaluation`, `partitioning`).
- Use `scripts/preflight_env.sh` to validate environment readiness.
