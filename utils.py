def print_elapsed_time(freq=15):
    import time  
    start_time = time.time()
    while True:
        time.sleep(freq)
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {int(elapsed_time)} seconds")