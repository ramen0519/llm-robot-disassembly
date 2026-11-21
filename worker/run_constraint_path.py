"""New extension entrypoint; the existing independent worker is unchanged."""
from contextlib import contextmanager
import inspect
import sys
import run_independent_rank as worker
from constraint_path_hook import ConstraintPathHook


def main():
    controlled = '--controlled-path-probe' in sys.argv
    if controlled:
        sys.argv.remove('--controlled-path-probe')
    original = worker.installed_independent

    @contextmanager
    def installed(generator, registry, planner, session):
        signature = inspect.signature(planner._simulate)
        hook = ConstraintPathHook(session, signature, controlled)
        with hook.installed(planner):
            with original(generator, registry, planner, session):
                yield
    worker.installed_independent = installed
    try:
        return worker.main()
    finally:
        worker.installed_independent = original


if __name__ == '__main__':
    raise SystemExit(main())
