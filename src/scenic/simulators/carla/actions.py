"""Actions for dynamic agents in CARLA scenarios."""

import math as _math


import sys
try:
	sys.path.append('CARLA_0.9.10/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg')
	sys.path.append('CARLA_0.9.10/PythonAPI/carla/')
	import carla as _carla
except ImportError as e:
	raise ModuleNotFoundError('CARLA scenarios require the "carla" Python package') from e


import bbox_annotation.carla_vehicle_annotator as cva
from agents.navigation.basic_agent import BasicAgent
from scenic.domains.driving.actions import *
import scenic.simulators.carla.utils.utils as _utils
import scenic.simulators.carla.model as _carlaModel
import pdb
import numpy as np
import os



################################################
# Actions available to all carla.Actor objects #
################################################

SetLocationAction = SetPositionAction	# TODO refactor

class SetAngularVelocityAction(Action):
	def __init__(self, angularVel):
		self.angularVel = angularVel

	def applyTo(self, obj, sim):
		xAngularVel = self.angularVel * _math.cos(obj.heading)
		yAngularVel = self.angularVel * _math.sin(obj.heading)
		newAngularVel = _utils.scalarToCarlaVector3D(xAngularVel, yAngularVel)
		obj.carlaActor.set_angular_velocity(newAngularVel)

class SetTransformAction(Action):	# TODO eliminate
	def __init__(self, pos, heading):
		self.pos = pos
		self.heading = heading

	def applyTo(self, obj, sim):
		loc = _utils.scenicToCarlaLocation(self.pos, z=obj.elevation)
		rot = _utils.scenicToCarlaRotation(self.heading)
		transform = _carla.Transform(loc, rot)
		obj.carlaActor.set_transform(transform)


#############################################
# Actions specific to carla.Vehicle objects #
#############################################

class VehicleAction(Action):
	def canBeTakenBy(self, agent):
		return isinstance(agent, _carlaModel.Vehicle)

class SetManualGearShiftAction(VehicleAction):
	def __init__(self, manualGearShift):
		if not isinstance(manualGearShift, bool):
			raise RuntimeError('Manual gear shift must be a boolean.')
		self.manualGearShift = manualGearShift

	def applyTo(self, obj, sim):
		vehicle = obj.carlaActor
		ctrl = vehicle.get_control()
		ctrl.manual_gear_shift = self.manualGearShift
		vehicle.apply_control(ctrl)


class SetGearAction(VehicleAction):
	def __init__(self, gear):
		if not isinstance(gear, int):
			raise RuntimeError('Gear must be an int.')
		self.gear = gear

	def applyTo(self, obj, sim):
		vehicle = obj.carlaActor
		ctrl = vehicle.get_control()
		ctrl.gear = self.gear
		vehicle.apply_control(ctrl)


class SetManualFirstGearShiftAction(VehicleAction):	# TODO eliminate
	def applyTo(self, obj, sim):
		ctrl = _carla.VehicleControl(manual_gear_shift=True, gear=1)
		obj.carlaActor.apply_control(ctrl)


class SetTrafficLightAction(VehicleAction):
	"""Set the traffic light to desired color. It will only take
	effect if the car is within a given distance of the traffic light.

	Arguments:
		color: the string red/yellow/green/off/unknown
		distance: the maximum distance to search for traffic lights from the current position
	"""
	def __init__(self, color, distance=100, group=False):
		self.color = _utils.scenicToCarlaTrafficLightStatus(color)
		if color is None:
			raise RuntimeError('Color must be red/yellow/green/off/unknown.')
		self.distance = distance

	def applyTo(self, obj, sim):
		traffic_light = obj._getClosestTrafficLight(self.distance)
		if traffic_light is not None:
			traffic_light.set_state(self.color)

class SetAutopilotAction(VehicleAction):
	def __init__(self, enabled):
		if not isinstance(enabled, bool):
			raise RuntimeError('Enabled must be a boolean.')
		self.enabled = enabled

	def applyTo(self, obj, sim):
		vehicle = obj.carlaActor
		vehicle.set_autopilot(self.enabled, sim.tm.get_port())

class SetVehicleLightStateAction(VehicleAction):
	"""Set the vehicle lights' states.

	Arguments:
		vehicleLightState: Which lights are on.
	"""
	def __init__(self, vehicleLightState):
		self.vehicleLightState = vehicleLightState

	def applyTo(self, obj, sim):
		obj.carlaActor.set_light_state(self.vehicleLightState)

#################################################
# Actions available to all carla.Walker objects #
#################################################

class PedestrianAction(Action):
	def canBeTakenBy(self, agent):
		return isinstance(agent, _carlaModel.Pedestrian)

class SetJumpAction(PedestrianAction):
	def __init__(self, jump):
		if not isinstance(jump, bool):
			raise RuntimeError('Jump must be a boolean.')
		self.jump = jump

	def applyTo(self, obj, sim):
		walker = obj.carlaActor
		ctrl = walker.get_control()
		ctrl.jump = self.jump
		walker.apply_control(ctrl)

class SetWalkAction(PedestrianAction):
	def __init__(self, enabled, maxSpeed=1.4):
		if not isinstance(enabled, bool):
			raise RuntimeError('Enabled must be a boolean.')
		self.enabled = enabled
		self.maxSpeed = maxSpeed

	def applyTo(self, obj, sim):
		controller = obj.carlaController
		if self.enabled:
			controller.start()
			controller.go_to_location(sim.world.get_random_location_from_navigation())
			controller.set_max_speed(self.maxSpeed)
		else:
			controller.stop()

class SetWalkDestination(PedestrianAction):
	def __init__(self,destination):
		self.destination = _utils.scenicToCarlaLocation(destination,z=0)
	def applyTo(self, obj, sim):
		controller = obj.carlaController
		controller.go_to_location(self.destination)
		

class TrackWaypointsAction(Action):
	def __init__(self, waypoints, cruising_speed = 10):
		self.waypoints = np.array(waypoints)
		self.curr_index = 1
		self.cruising_speed = cruising_speed
		pdb.set_trace()

	def canBeTakenBy(self, agent):
		# return agent.lgsvlAgentType is lgsvl.AgentType.EGO
		return True

	def LQR(v_target, wheelbase, Q, R):
		A = np.matrix([[0, v_target*(5./18.)], [0, 0]])
		B = np.matrix([[0], [(v_target/wheelbase)*(5./18.)]])
		V = np.matrix(linalg.solve_continuous_are(A, B, Q, R))
		K = np.matrix(linalg.inv(R)*(B.T*V))
		return K

	def applyTo(self, obj, sim):
		carlaObj = obj.carlaActor
		transform = carlaObj.get_transform()
		pos = transform.location
		rot = transform.rotation
		velocity = carlaObj.get_velocity()
		th, x, y, v = rot.pitch/180.0*np.pi, pos.x, pos.z, (velocity.x**2 + velocity.z**2)**0.5
		#print('state:', th, x, y, v)
		PREDICTIVE_LENGTH = 3
		MIN_SPEED = 1
		WHEEL_BASE = 3
		v = max(MIN_SPEED, v)

		x = x + PREDICTIVE_LENGTH * np.cos(-th+np.pi/2)
		y = y + PREDICTIVE_LENGTH * np.sin(-th+np.pi/2)
		#print('car front:', x, y)
		dists = np.linalg.norm(self.waypoints - np.array([x, y]), axis=1)
		dist_pos = np.argpartition(dists,1)
		index = dist_pos[0]
		if index > self.curr_index and index < len(self.waypoints)-1:
			self.curr_index = index
		p1, p2, p3 = self.waypoints[self.curr_index-1], self.waypoints[self.curr_index], self.waypoints[self.curr_index+1]

		p1_a = np.linalg.norm(p1 - np.array([x, y]))
		p3_a = np.linalg.norm(p3 - np.array([x, y]))
		p1_p2= np.linalg.norm(p1 - p2)
		p3_p2= np.linalg.norm(p3 - p2)

		if p1_a - p1_p2 > p3_a - p3_p2:
			p1 = p2
			p2 = p3

		#print('points:',p1, p2)
		x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
		th_n = -math.atan2(y2-y1,x2-x1)+np.pi/2
		d_th = (th - th_n + 3*np.pi) % (2*np.pi) - np.pi
		d_x = (x2-x1)*y - (y2-y1)*x + y2*x1 - y1*x2
		d_x /= np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
		#print('d_th, d_x:',d_th, d_x)


		K = TrackWaypoints.LQR(v, WHEEL_BASE, np.array([[1, 0], [0, 3]]), np.array([[10]]))
		u = -K * np.matrix([[-d_x], [d_th]])
		u = np.double(u)
		u_steering = min(max(u, -1), 1)

		K = 1
		u = -K*(v - self.cruising_speed)
		u_thrust = min(max(u, -1), 1)

		#print('u:', u_thrust, u_steering)

		ctrl = carlaObj.get_control()
		ctrl.steering = u_steering
		if u_thrust > 0:
			ctrl.throttle = u_thrust
		elif u_thrust < 0.1:
			ctrl.braking = -u_thrust
		carlaObj.apply_control(ctrl)
		
		
class GetPathVehicle(Action):


	def __init__(self,obj,actor,sim):
		self.position = actor.carlaActor.get_transform().location
		self.agent = BasicAgent(obj.carlaActor)
		map = sim.map
		
		chosen_waypoint = map.get_waypoint(self.position,project_to_road=True, lane_type=_carla.LaneType.Driving)
		current_waypoint = map.get_waypoint(obj.carlaActor.get_transform().location,project_to_road=True, lane_type=_carla.LaneType.Driving)
		new_route_trace = self.agent._trace_route(current_waypoint, chosen_waypoint)
		#pdb.set_trace()
		print(new_route_trace)
		self.agent._local_planner.set_global_plan(new_route_trace)
		obj.carlaActor.apply_control(self.agent.run_step())
		self.counter = 0
		
		
	def applyTo(self,obj,sim):
		
		print("k22")
		if not self.counter:	
			
			self.counter += 1
			print("aqui")
		else:
			print("aqui2")
			control = self.agent._local_planner.run_step()
			obj.carlaActor.apply_control(control)


import socket
from threading import Thread
import threading
import cv2
class SendImages(Action):

	def __init__(self, frame_index, camera_id, server_connection, current_server_listening_thread, stop_listening_event, path):
		self.camera_id = camera_id
		self.frame_index = frame_index
		self.server_connection = server_connection
		self.current_server_listening_thread = current_server_listening_thread
		self.stop_listening_event = stop_listening_event
		self.path = path

	def mysend(self, msg, MSGLEN):
        	totalsent = 0
        	while totalsent < MSGLEN:
        	    sent = self.server_connection.send(msg[totalsent:])
        	    if sent == 0:
        	        raise RuntimeError("socket connection broken")
        	    totalsent = totalsent + sent
        	    
	def get_camera_intrinsic(self):
		VIEW_WIDTH = 800
		VIEW_HEIGHT = 600
		VIEW_FOV = 90
		calibration = np.identity(3)
		calibration[0, 2] = VIEW_WIDTH / 2.0
		calibration[1, 2] = VIEW_HEIGHT / 2.0
		calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
		return calibration
		
	def get_matrix(self, transform):
		rotation = transform.rotation
		location = transform.location
		c_y = np.cos(np.radians(rotation.yaw))
		s_y = np.sin(np.radians(rotation.yaw))
		c_r = np.cos(np.radians(rotation.roll))
		s_r = np.sin(np.radians(rotation.roll))
		c_p = np.cos(np.radians(rotation.pitch))
		s_p = np.sin(np.radians(rotation.pitch))
		matrix = np.matrix(np.identity(4))
		matrix[0, 3] = location.x
		matrix[1, 3] = location.y
		matrix[2, 3] = location.z
		matrix[0, 0] = c_p * c_y
		matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
		matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
		matrix[1, 0] = s_y * c_p
		matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
		matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
		matrix[2, 0] = s_p
		matrix[2, 1] = -c_p * s_r
		matrix[2, 2] = c_p * c_r
		return matrix    
    
	def applyTo(self, obj, sim): #We should put into a thread this
		
		if obj.cam_queue: 
		
			# Get images
			rgb_image = obj.cam_queue[-1] #.get()
			# Make sure the image queues stay under the size
			obj.cam_queue.clear()
			array = np.frombuffer(rgb_image.raw_data, dtype=np.dtype("uint8"))
			
			
			if self.path:
				if not os.path.exists(self.path):
					os.makedirs(self.path)
				if not os.path.exists(self.path+'/'+str(self.camera_id)):
					os.makedirs(self.path+'/'+str(self.camera_id))
				rgb_image.save_to_disk("%s/%s/%05d.jpg" % (self.path,self.camera_id,rgb_image.frame))
			#pdb.set_trace()
			
			'''
			K = self.get_camera_intrinsic()
			Rt = self.get_matrix(rgb_image.transform)[:3]
			F = np.array([[ 0,  1,  0 ], [ 0,  0, -1 ], [ 1,  0,  0 ]], dtype=np.float32)
			
			

			vector_3d = np.array([0.786292,5.856472,0.959821,1])
			FRt = np.matmul(F,Rt)
			P = np.matmul(K,FRt)
			
			res = np.array(np.matmul(P,vector_3d))
			print(res/res[0][2])
			
			K_inv = np.linalg.inv(K)
			'''
			
			
			arr_bytes = array.tobytes()
			try:
			
				#print("Sending data",self.frame_index)
				
				#len_bytes = self.server_connection.send(self.frame_index.to_bytes(2, 'big'))
				self.mysend(self.frame_index.to_bytes(2, 'big'),2)

				#print("Sent bytes", self.frame_index)
				#len_bytes = self.server_connection.send(arr_bytes)
				self.mysend(arr_bytes,len(arr_bytes))
				#print("Sent Frame: " + str(self.frame_index), self.frame_index)
				self.frame_index += 1
				# time.sleep(0.5)
			except Exception as e:
				print("Socket timeout!", e)
				obj.connected = False
				#self.server_connection = self.setup_connections_and_handling()
				#self.stop_listening_event.set() # This will stop the thread
				#self.current_server_listening_thread.join()
				# Set up our listener again
				#self.stop_listening_event.clear()
				#self.current_server_listening_thread = self.setupListeningServer()
				


		
class GetBoundingBox(Action):
	"""Get bounding boxes"""
	def __init__(self, actors, path):
		self.actors = actors
		self.path = path
		#print(len(actors))
	def applyTo(self, obj, sim):
		if obj.depth.cam_queue and obj.cam_queue:  # We only save information if we have captured data

			#desc = camqueue[2]
			# print(desc)

			# Get images
			depth_image = obj.depth.cam_queue[-1]
			rgb_image = obj.cam_queue[-1] #.get()
			# Make sure the image queues stay under the size
			obj.depth.cam_queue.clear()
			obj.cam_queue.clear()

			# Save RGB image with bounding boxes
			depth_meter = cva.extract_depth(depth_image)
			#filtered, removed = cva.auto_annotate(vehicle_actors, \
			#    camera_actors[i][0], depth_meter)
			#pdb.set_trace()
			actors_bb = [x.carlaActor if isinstance(x.carlaActor,_carla.Walker) or isinstance(x.carlaActor,_carla.Vehicle) else x for x in self.actors if x.carlaActor is not None]
			#pdb.set_trace()
			filtered, removed = cva.auto_annotate(actors_bb,obj.carlaActor, depth_meter, depth_margin=100)
			#pdb.set_trace()
			# Get the corresponding metadata for these vehicles
			metadata = []

			#print(filtered["vehicles"], object_actors)
			for filtered_vehicle in filtered["vehicles"]:


				vehicle_traffic_state = "N/A"

				if (isinstance(filtered_vehicle,_carla.libcarla.Walker)):
					#pdb.set_trace()
					metadata_entry = ["pedestrian",filtered_vehicle.attributes,filtered_vehicle.type_id,filtered_vehicle.id]
				elif (isinstance(filtered_vehicle,_carla.libcarla.Vehicle)):
					metadata_entry = ["vehicle",filtered_vehicle.attributes,filtered_vehicle.type_id, filtered_vehicle.id]
				else:
					metadata_entry = ["object",filtered_vehicle.carlaActor.attributes,filtered_vehicle.carlaActor.type_id,filtered_vehicle.carlaActor.id]

	


				'''
				# Get current vehicle attributes
				vehicle_loc = filtered_vehicle.get_transform().location
				vehicle_rot = filtered_vehicle.get_transform().rotation
				vehicle_acc = filtered_vehicle.get_acceleration()
				vehicle_vel = filtered_vehicle.get_velocity()
				vehicle_ang_vel = filtered_vehicle.get_angular_velocity()



				current_vehicle_attributes = {}
				current_vehicle_attributes["location"] = [vehicle_loc.x, \
				    vehicle_loc.y, vehicle_loc.z]
				current_vehicle_attributes["rotation"] = [vehicle_rot.pitch, \
				    vehicle_rot.yaw, vehicle_rot.roll]
				current_vehicle_attributes["acceleration"] = [vehicle_acc.x, \
				    vehicle_acc.y, vehicle_acc.z]
				current_vehicle_attributes["velocity"] = [vehicle_vel.x, \
				    vehicle_vel.y, vehicle_vel.z]
				current_vehicle_attributes["angular_velocity"] = [vehicle_ang_vel.x, \
				    vehicle_ang_vel.y, vehicle_ang_vel.z]
				current_vehicle_attributes["traffic_state"] = vehicle_traffic_state


				metadata_entry.append(current_vehicle_attributes)
				'''

				metadata.append(metadata_entry)

			# We only save the image under two conditions:
			#   - There is actually a vehicle in the image
			#   - For a single time per simulation, we save the first image that has no vehicles
			# if metadata: or not stored_empty_image:


			cva.save_output(rgb_image, filtered['bbox'], filtered['class'], \
			removed['bbox'], removed['class'], path=self.path+"tc"+str(obj.camera_id), \
			save_patched=True, add_data=metadata, out_format='json')

