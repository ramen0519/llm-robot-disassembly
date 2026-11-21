"""Independent LLM candidates and natural-failure repair; no controlled probe."""
from contextlib import contextmanager
import inspect
import run_independent_rank as worker
from integrated_path_hook import IntegratedPathHook


def main():
    original = worker.installed_independent

    @contextmanager
    def installed(generator, registry, planner, session):
        hook = IntegratedPathHook(session, inspect.signature(planner._simulate))
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
