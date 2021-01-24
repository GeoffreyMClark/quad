import numpy as np
import pybullet as p
import pybullet_data as pd
import time
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os


def buildIKModel():
    checkpoint_path = "training/ik_1.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model_ik = tf.keras.Sequential([
        layers.Dense(50, activation=tf.nn.relu, input_shape=[3]),
        layers.Dense(50, activation=tf.nn.relu),
        layers.Dense(50, activation=tf.nn.relu),
        layers.Dense(3)
    ])
    model_ik.compile(optimizer='adam', loss='mae')
    model_ik.summary()
    # Loads the weights
    model_ik.load_weights(checkpoint_path)
    return model_ik

def buildFKModel():
    checkpoint_path = "training/fk_1.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model_fk = tf.keras.Sequential([
        layers.Dense(50, activation=tf.nn.relu, input_shape=[3]),
        layers.Dense(50, activation=tf.nn.relu),
        layers.Dense(50, activation=tf.nn.relu),
        layers.Dense(3)
    ])
    model_fk.compile(optimizer='adam', loss='mae')
    model_fk.summary()
    # Loads the weights
    model_fk.load_weights(checkpoint_path)
    return model_fk

def current_milli_time():
    return round(time.time(),6)

class cheetah_class:
    position = np.array([0.0,0.0,0.0])
    orientation = np.array([0.0,0.0,0.0])
    joint_angles = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    joint_vels = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    joint_accs = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

class goal_class:
    position = np.array([0.0,0.0,0.25])
    orientation = np.array([0.0,0.0,0.0])
    leg_error = np.array([0.0,0.0,0.0,0.0])

def add_robot():
    p.setGravity(0,0,-9.8)
    p.setAdditionalSearchPath(pd.getDataPath())
    floor = p.loadURDF("plane.urdf")
    startPos = [0,0,0.5]
    robot = p.loadURDF("mini_cheetah.urdf",startPos) #,flags=p.URDF_USE_SELF_COLLISION|p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT
    numJoints = p.getNumJoints(robot)
    for j in range (-1,numJoints):
        p.changeVisualShape(robot,j,rgbaColor=[1,1,1,1])
    pos0=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    robot_position_ctrl(robot,numJoints,pos0,200)
    # zero robot on ground for two seconds
    for i in range(1*240):
        pos=np.array([15.0,-90.0,160.0,0.0,15.0,-90.0,160.0,0.0,15.0,-90.0,160.0,0.0,15.0,-90.0,160.0,0.0])
        robot_position_ctrl(robot,numJoints,pos,200)
        p.stepSimulation()
        time.sleep(dt)
    # pos=np.array([15.0,-90.0,160.0,0.0,15.0,-90.0,160.0,0.0,15.0,-90.0,160.0,0.0,15.0,-90.0,160.0,0.0])
    # robot_position_ctrl(robot,numJoints,pos,0)
    return robot,numJoints

def robot_position_ctrl(robot,numJoints,pos,force):
    pos_adj = np.array([-1,1,1,1,1,1,1,1,-1,1,1,1,1,1,1,1])
    pos_ctrl = pos*pos_adj*np.pi/180
    for j in range(numJoints):
        p.setJointMotorControl2(robot,j,p.POSITION_CONTROL,pos_ctrl[j],0,force=force)

def robot_torque_control(robot,torque_ctrl):
    for j in [1,2,5,6,9,10,13,14]:
        p.setJointMotorControl2(robot,j,p.TORQUE_CONTROL,force=torque_ctrl[j])


def robot_data(robot, robot_state):
    [location, quaternions] = p.getBasePositionAndOrientation(robot)
    robot_state.position = np.asarray(location)
    robot_state.orientation = np.asarray(p.getEulerFromQuaternion(quaternions))
    joints = p.getJointStates(robot,range(numJoints))
    robot_state.joint_angles = np.array([state[0] for state in joints])*180/np.pi
    joint_vels = np.array([state[1] for state in joints])*180/np.pi
    joint_torques = np.array([state[3] for state in joints])
    robot_state.joint_accs = (robot_state.joint_vels-joint_vels)/500
    robot_state.joint_vels = joint_vels


def standing_control(robot, numjoints, robot_state, goal_state, model_ik, model_fk, dt):
    # Calculate body errors
    position_error = goal_state.position[2]-robot_state.position[2]
    # print("Position Error - ",position_error)
    orientation_error = goal_state.orientation-robot_state.orientation
    roll_error = np.sin(orientation_error[0])*0.111
    pitch_error = np.sin(orientation_error[1])*0.19
    # Calculate forward kinematics from model
    current_angles = np.concatenate((robot_state.joint_angles[0:3].reshape(1,3), robot_state.joint_angles[4:7].reshape(1,3), robot_state.joint_angles[8:11].reshape(1,3), robot_state.joint_angles[12:15].reshape(1,3)),axis=0)
    forward_positions = model_fk(current_angles.reshape(4,3),training=False).numpy()
    # Translate body errors to individual leg errors
    Pgain = 0.6666; Dgain = -0.1
    leg1_error =  (position_error-roll_error-pitch_error)
    leg2_error =  (position_error+roll_error-pitch_error)
    leg3_error =  (position_error-roll_error+pitch_error)
    leg4_error =  (position_error+roll_error+pitch_error)
    # Position P Controller
    leg1_P = leg1_error*Pgain
    leg2_P = leg2_error*Pgain
    leg3_P = leg3_error*Pgain
    leg4_P = leg4_error*Pgain
    # Derivitive D controller
    leg1_D = ((goal_state.leg_error[0]-leg1_error)/dt)*Dgain
    leg2_D = ((goal_state.leg_error[1]-leg2_error)/dt)*Dgain
    leg3_D = ((goal_state.leg_error[2]-leg3_error)/dt)*Dgain
    leg4_D = ((goal_state.leg_error[3]-leg4_error)/dt)*Dgain
    print(leg1_D)
    goal_state.leg_error[0] = leg1_error
    goal_state.leg_error[1] = leg2_error
    goal_state.leg_error[2] = leg3_error
    goal_state.leg_error[3] = leg4_error
    # Control Output
    leg1_position = (forward_positions[0][1]+leg1_P+leg1_D)
    leg2_position = (forward_positions[1][1]+leg2_P+leg2_D)
    leg3_position = (forward_positions[2][1]+leg3_P+leg3_D)
    leg4_position = (forward_positions[3][1]+leg4_P+leg4_D)
    control_positions = np.array([[0, leg1_position, 0],[0, leg2_position, 0],[0, leg3_position, 0],[0, leg4_position, 0]])
    # query model to get foot extension
    time1 = current_milli_time()
    control_angles = model_ik(control_positions.reshape(4,3),training=False).numpy()
    time3 = current_milli_time()
    # print("time - ", time3-time1)
    # Send joint angles to robot for control with max force
    l11 = control_angles[0,0];     l12=control_angles[0,1];     l13=control_angles[0,2]
    l21 = control_angles[1,0];     l22=control_angles[1,1];     l23=control_angles[1,2]
    l31 = control_angles[2,0];     l32=control_angles[2,1];     l33=control_angles[2,2]
    l41 = control_angles[3,0];     l42=control_angles[3,1];     l43=control_angles[3,2]
    pos=np.array([l11,l12,l13,0.0,l21,l22,l23,0.0,l31,l32,l33,0.0,l41,l42,l43,0.0])
    robot_position_ctrl(robot,numJoints,pos,10)
    pass
 


if __name__ =="__main__":
    # initialize world
    p.connect(p.GUI)
    dt = 1./200.

    # Add robot
    robot,numJoints = add_robot()
    robot_state = cheetah_class()
    goal_state = goal_class()
    robot_data(robot, robot_state)
    model_ik = buildIKModel()
    model_fk = buildFKModel()

    while(1):
        # Robot observation
        robot_data(robot, robot_state)
        # print(robot_state.position)

        # Impediance controller (convert R to P joints)
        standing_control(robot, numJoints, robot_state, goal_state, model_ik, model_fk, dt)

        # Take simulation steps
        p.stepSimulation()
        time.sleep(dt)
		






