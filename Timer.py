import time
def timer(func):
    def wrapper(function_name, *args, **kwargs):
        p_t = time.time()
        result = func(*args, **kwargs)
        c_t = time.time()
        dt = c_t - p_t
        print(f"the time taken by {function_name} is {dt} seconds.")

        return result
    return wrapper