import sys
import inspect
import time

import tests.test_training_invariants as tti

def main():
    test_funcs = [obj for name, obj in inspect.getmembers(tti) if name.startswith('test_') and inspect.isfunction(obj)]
    print(f"Discovered {len(test_funcs)} invariant tests.")
    passed = 0
    failed = 0
    for i, fn in enumerate(test_funcs, 1):
        t0 = time.time()
        print(f"[{i:02d}/{len(test_funcs):02d}] {fn.__name__}...", end=" ", flush=True)
        try:
            fn()
            print(f"PASSED ({time.time()-t0:.2f}s)")
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_funcs)} tests.")
    if failed > 0:
        sys.exit(1)

if __name__ == '__main__':
    main()
