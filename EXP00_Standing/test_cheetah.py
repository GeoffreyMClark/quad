import numpy as np
import pybullet as p
import pybullet_data as pd
import time
import math
import tinyik

# actuator design for individual leg
leg = tinyik.Actuator([[-.49, .0, 1.9], 'z', [-.62, .0, .0], 'x', [.0, -2.09, .0], 'x', [.0, -1.8, .0]])

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


def add_robot():
	# Set gravity
	p.setGravity(0,0,-9.8)
	# Generate floor
	p.setAdditionalSearchPath(pd.getDataPath())
	floor = p.loadURDF("plane.urdf")
	# Generate robot
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


def standing_controller(robot, numjoints, robot_state, goal_state):
	# Calculate body errors
	position_error = goal_state.position-robot_state.position
	orientation_error = goal_state.orientation-robot_state.orientation
	# Translate body errors to individual leg errors
	leg1_error = position_error
	# query direct kinematics to get foot extension
	time1 = current_milli_time()
	leg.ee = [-1.11, -(robot_state.position[2]+position_error[2])*10, 1.6]
	l1,l2,l3 = np.rad2deg(leg.angles)
	time3 = current_milli_time()
	# convert IK frame to robot frame
	l1 = l1;     l2=-l2;     l3=-l3
	# Send joint angles to robot for control with max force
	pos=np.array([l1,l2,l3,0.0,l1,l2,l3,0.0,l1,l2,l3,0.0,l1,l2,l3,0.0])
	robot_position_ctrl(robot,numJoints,pos,5)
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

	while(1):
		# Robot observation
		robot_data(robot, robot_state)
		print(robot_state.position)
		leg.angles = np.deg2rad([robot_state.joint_angles[0], -robot_state.joint_angles[1], -robot_state.joint_angles[2]])

		# Impediance controller (convert R to P joints)
		standing_controller(robot, numJoints, robot_state, goal_state)

		# Take simulation steps
		p.stepSimulation()
		time.sleep(dt)
		






