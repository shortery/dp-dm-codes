def cause_sigsevg():
    import ctypes

    ctypes.string_at(0)


def my_fn(should_crash: bool = False):
    if should_crash:
        cause_sigsevg()
    return 42


# create a new process and execute my_fn
# make it safe in case of segfault
if __name__ == "__main__":
    import concurrent.futures
    import sys

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future = executor.submit(my_fn, True)
        try:
            result = future.result()
        except Exception as e:
            print(f"Caught exception: {repr(e)}")
        else:
            print(f"Result: {result}")
