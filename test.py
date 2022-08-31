from simple_pid import PID
import time
import numpy as np
dt = 1./240

pid = PID(10, 5, 0.5, setpoint=np.array([1.0,1]))
pid.sample_time = dt

cur_value = np.array([0.0,0.0])
tgt_value = np.array([1.0,1])
time_start = time.time()

while True:
    tgt_value = np.array([1.0,1])#np.sin(1 * (time.time() - time_start))
    pid.setpoint = tgt_value
    output = pid(cur_value)
    cur_value += dt * output
    print("cur_value =", cur_value, ",   tgt_value =", tgt_value, ",   output =", output)
    time.sleep(dt)