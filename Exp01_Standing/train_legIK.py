import tinyik
import numpy as np
import time
import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

def current_milli_time():
        return round(time.time(),6)

def save_data_in_csv(positions, angles, num_samples,file_name):
    csv1 = open(file_name, "a")
    for i in range(num_samples):
        csv1.write(str(positions[i,0]) + "," + str(positions[i,1]) + "," + str(positions[i,2]) + "," + str(angles[i,0]) + "," + str(angles[i,1]) + "," + str(angles[i,2]) + "\n")
    csv1.close()

def read_data_in_csv(file_name):
    with open(file_name) as csv_file:
        data = np.asarray(list(csv.reader(csv_file, delimiter=',')), dtype=np.float32)
        positions = data[:,0:3]
        angles = data[:,3:6]
    return positions,angles



def generate_data(num_samples,file_name):
    leg = tinyik.Actuator([[-.049, .0, .19], 'z', [-.062, .0, .0], 'x', [.0, -.209, .0], 'x', [.0, -.18, .0]])
    x_out = np.random.uniform(.11,2.11,num_samples)
    y_down = np.random.uniform(.3,3,num_samples)
    z_forward = np.random.uniform(0,3.2,num_samples)
    positions = np.concatenate((x_out.reshape(-1,1), y_down.reshape(-1,1), z_forward.reshape(-1,1)),axis=1)

    angles = np.empty([0,3])
    time1 = current_milli_time()
    for i in range(num_samples):
        leg.angles = np.deg2rad([6.06137964e-07, 8.39262632e+01, 1.94986044e+02])
        leg.ee = [-x_out[i], -y_down[i], z_forward[i]]
        # tinyik.visualize(leg)
        angles = np.concatenate((angles,np.rad2deg(leg.angles).reshape(1,3)),axis=0)
        # Convert IK to robot model
        angles[0,1] = -angles[0,1]; angles[0,2] = 360-angles[0,2]
        print(i)
    time2 = current_milli_time()
    print('final time - ',time2-time1)
    save_data_in_csv(positions, angles,num_samples,file_name)
    return positions, angles


def buildModel():
    model = tf.keras.Sequential([
        layers.Dense(50, activation=tf.nn.relu, input_shape=[3]),
        layers.Dense(50, activation=tf.nn.relu),
        layers.Dense(50, activation=tf.nn.relu),
        layers.Dense(3)
    ])
    # model.compile(optimizer='adam', loss='mean_squared_error')
    model.compile(optimizer='adam', loss='mae')
    model.summary()
    return model





if __name__ =="__main__":
    # generate_data(10000,"quad_IK.csv")
    positions, angles = read_data_in_csv("quad_IK_1.csv")
    model = buildModel()
    EPOCHS = 100
    checkpoint_path = "training/cp_1.ckpt"; checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
    model.fit(positions, angles,epochs=EPOCHS, validation_split = 0.2, verbose=1, callbacks=[cp_callback])

    leg = tinyik.Actuator([[-.049, .0, .19], 'z', [-.062, .0, .0], 'x', [.0, -.209, .0], 'x', [.0, -.18, .0]])
    control_positions = np.array([[0, .25, 0],[0, .25, 0],[0, .25, 0],[0, .25, 0]])
    leg.ee = [1.11, .25, 1.6]
    for i in range(10):
        print("-------------------------------------------------------------")
        print("-------------------------------------------------------------")
        time1 = current_milli_time()
        control_angles = model(control_positions.reshape(4,3),training=False).numpy()
        # print(test_angles)
        # print(angles[0:4,:])
        time2 = current_milli_time()
        print('final time - ',time2-time1)



