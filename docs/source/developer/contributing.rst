Contributing
============

Baseline contribution flow:

1. Create a branch for your change.
2. Install dev dependencies:

.. code-block:: bash

   pip install -e ".[dev,docs]"

3. Run quick validation before opening a PR:

.. code-block:: bash

   ./scripts/preflight_env.sh

4. Add/adjust tests for non-trivial behavior changes.
5. Update docs when public interfaces or workflows change.
