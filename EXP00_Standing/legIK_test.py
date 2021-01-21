import tinyik
import numpy as np
import time

def current_milli_time():
    return round(time.time(),6)

leg = tinyik.Actuator([[-.49, .0, 1.9], 'z', [-.62, .0, .0], 'x', [.0, -2.09, .0], 'x', [.0, -1.8, .0]])
leg.angles = np.deg2rad([-15, 0, 0])
print(leg.ee)
tinyik.visualize(leg)


# time1 = current_milli_time()
# control_angles = model(control_positions.reshape(4,3),training=False).numpy()
# time3 = current_milli_time()
# print("time - ", time3-time1)

for i in [-.1,-.5,-1,-1.5,-2, -2.5]:
    # leg.ee = [-1.11, i, 1.6]
    time1 = current_milli_time()
    leg.angles = np.deg2rad([-15, i, 0])
    a=leg.ee
    time3 = current_milli_time()
    print("time - ", time3-time1)
    # print(np.rad2deg(leg.angles))
    # tinyik.visualize(leg)