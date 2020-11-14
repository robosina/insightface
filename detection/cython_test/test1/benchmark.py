from adder import collector

import time

for i in range(10):
    count = 1000000
    t1 = time.time()
    value = collector(count)
    print("time elapsed:{} and value is {}".format(time.time() - t1, value))

    from adder2 import collector as cl

    t1 = time.time()
    value = cl(count)
    print("time elapsed C:{} and value is {}".format(time.time() - t1, value))
