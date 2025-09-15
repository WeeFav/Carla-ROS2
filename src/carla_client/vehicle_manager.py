import carla
import random
import numpy as np

class VehicleManager():
    def __init__(self, world, tm, fps, respawn, num_vehicles):
        self.world = world
        self.tm = tm

        self.tm.set_hybrid_physics_mode(True)
        self.tm.set_respawn_dormant_vehicles(True)

        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()

        self.respawn_interval = respawn * fps
        self.num_vehicles = num_vehicles


    def spawn_ego_vehicle(self, autopilot):
        # Spawn ego vehicle
        bp_ego_vehicle = self.blueprint_library.find('vehicle.ford.mustang')
        bp_ego_vehicle.set_attribute('role_name', 'hero')
        self.ego_vehicle = self.world.spawn_actor(bp_ego_vehicle, random.choice(self.spawn_points))
        self.ego_vehicle.set_autopilot(autopilot, self.tm.get_port())
        self.tm.update_vehicle_lights(self.ego_vehicle, True)
        return self.ego_vehicle
    

    def spawn_vehicles(self):
        # Spawn vehicles
        self.vehicle_blueprints = self.blueprint_library.filter('*vehicle*')
        self.vehicle_state = {}
        
        i = 0
        while i < self.num_vehicles:
            vehicle = self.world.try_spawn_actor(random.choice(self.vehicle_blueprints), random.choice(self.spawn_points))
            if vehicle is not None:
                vehicle.set_autopilot(True, self.tm.get_port())
                self.tm.update_vehicle_lights(vehicle, True)
                i += 1


    def get_vehicle_state(self, vehicle):
        t = vehicle.get_transform().location
        v = vehicle.get_velocity()
        return (round(t.x, 2), round(t.y, 2), round(t.z, 2),
                round(v.x, 2), round(v.y, 2), round(v.z, 2))


    def check_vehicle(self):
        print("Checking vehicle states...")
        vehicles = self.world.get_actors().filter("vehicle.*")
        
        for vehicle in vehicles:
            # Skip ego vehicle
            role_name = vehicle.attributes.get("role_name", "")
            if role_name == "hero":
                continue

            current_state = self.get_vehicle_state(vehicle)
            if vehicle.id in self.vehicle_state:
                if current_state == self.vehicle_state[vehicle.id]:
                    print(f"[STUCK] Vehicle {vehicle.id} hasn't moved. Respawning...")
                    vehicle.destroy()

                    # Respawn at a random valid point
                    while True:
                        new_vehicle = self.world.try_spawn_actor(random.choice(self.vehicle_blueprints), random.choice(self.spawn_points))
                        if new_vehicle:
                            new_vehicle.set_autopilot(True, self.tm.get_port())
                            self.tm.update_vehicle_lights(new_vehicle, True)
                            self.vehicle_state[new_vehicle.id] = self.get_vehicle_state(new_vehicle)
                            break
                else:
                    self.vehicle_state[vehicle.id] = current_state
            else:
                self.vehicle_state[vehicle.id] = current_state


    def destroy(self):
        # Destroy all actors after game ends
        for vehicle in self.world.get_actors().filter('*vehicle*'):
            vehicle.destroy()
        print("All vehicles destroyed")


    def get_ego_vehicle_wheel(self):
        """
        return wheelbase, rear_axle_offset in vehicle frame
        """
        world_to_vehicle_matrix = np.array(self.ego_vehicle.get_transform().get_inverse_matrix()) 

        [front_left_world, front_right_world, back_left_world, back_right_world] = self.ego_vehicle.get_physics_control().wheels

        front_left_world = np.array([front_left_world.position.x / 100, front_left_world.position.y / 100, front_left_world.position.z / 100, 1]) # wheels are in cm in world frame
        back_left_world = np.array([back_left_world.position.x / 100, back_left_world.position.y / 100, back_left_world.position.z / 100, 1])

        front_left_world = world_to_vehicle_matrix @ front_left_world
        back_left_world = world_to_vehicle_matrix @ back_left_world

        front_left_vehicle = front_left_world[:3]  # x, y, z in vehicle frame
        back_left_vehicle = back_left_world[:3]
        
        wheelbase = front_left_vehicle[0] - back_left_vehicle[0]
        rear_axle_offset = -back_left_vehicle[0]

        return wheelbase, rear_axle_offset

