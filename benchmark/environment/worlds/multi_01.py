from benchmark.environment.world import World, ControllerConfig

from benchmark.environment.car_controller import CarController
from benchmark.environment.ped_controller import *

import carla


class WorldMulti01(World):
    def __init__(self, carla_world, hud, scenario, args):
        super().__init__(carla_world, hud, scenario, args)
        self.debug = True

    def tick_w1(self):
        status = self.path_controller_1.step()
        self.turn_head_right_behind.step()
        self.iss_crossed.step()
        if status == "Done":
            self.reset_pose.step()
            if self.choice is None:
                self.path_controller_2.step()
                distance = y_distance(self.walker.get_location(), self.player.get_location()) - 2
                if self.decision_trigger(distance, self.db, without_speed=True):
                    self.choice = "Wait"
                    self.walker.icr = ICR.PLANNING_TO
                    self.walker.son = SON.YIELDING
                else:
                    self.choice = "Cross"
                    self.walker.icr = ICR.GOING_TO
                    self.walker.son = SON.FORCING
            elif self.choice == "Cross":
                self.path_controller_2.step()
            elif self.choice == "Wait":
                self.path_controller_3.step()
                # Note: db is updated each tick
                distance = y_distance(self.walker.get_location(), self.player.get_location()) - 2
                if not self.decision_trigger(distance, self.db):
                    self.choice = "Cross"
                    self.walker.icr = ICR.GOING_TO
                    self.walker.son = SON.FORCING

    def tick_w2(self):
        self.path_controller_w2.step()
        self.w2_raise_arm.step()

    def tick(self, clock):
        super().tick(clock)
        self.update_db()
        self.tick_w1()
        self.tick_w2()

    def update_db(self):
        crossing_width = 9  # 9 m
        ped_crossing_time = crossing_width / self.ped_crossing_speed  # in m / m/s = s
        car_speed = self.player.get_velocity().length()  # in m/s

        # print("Ped cross time", ped_crossing_time)
        # print("Car speed: ", car_speed)
        car_travel_distance = car_speed * ped_crossing_time
        self.db = [car_travel_distance / 4 if self.walker.initial_son == SON.FORCING else 0, car_travel_distance]
        # if self.debug:
        #     self._draw_db()

    def setup_w1(self, blueprint, spawn_transform: carla.Transform, conf):
        self.ped_crossing_speed = self.ped_speed * 1.0 if conf.char == "yielding" else 1.1 * 1.1 * 1.1
        spawn_transform.location.y -= conf.spawning_distance
        self.walker = self.world.try_spawn_actor(blueprint, spawn_transform)

        self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), self.ped_speed))
        self.walker.icr = ICR.LOW
        self.walker.son = SON.FORCING if conf.char == "forcing" else SON.YIELDING
        self.walker.initial_son = SON.YIELDING if conf.char == "yielding" else SON.FORCING

        # Walk along the sidewalk and to the curb
        offsets_1 = [(0, 10), (1, 11)]
        self.path_1 = self._compute_plans(
            offsets_1, spawn_transform.location, color=carla.Color(r=255, g=0, b=0) if self.debug else None
        )
        self.path_controller_1 = PathController(self.world, self.walker, self.path_1, self.ped_speed)

        self.turn_head_right_behind = TurnHeadRightBehind(walker=self.walker, start_pos=self.path_1[0])
        self.reset_pose = ResetPose(self.walker)
        # Walk to the other side of the road
        offsets_2 = [(11, 11 + conf.crossing_distance), (12, 20)]
        self.path_2 = self._compute_plans(
            offsets_2, spawn_transform.location, color=carla.Color(r=0, g=255, b=0) if self.debug else None
        )
        self.path_controller_2 = PathController(self.world, self.walker, self.path_2, self.ped_crossing_speed)

        self.path_controller_3 = PathController(self.world, self.walker, self.path_2, 0)

        self.iss_crossed = InternalStateSetter(self.walker, self.path_2[1], icr=ICR.VERY_LOW, son=SON.AVERTING)

    def setup_w2(self, blueprint, spawn_transform: carla.Transform, conf):
        spawn_transform.location.y -= conf.spawning_distance
        self.walker2 = self.world.try_spawn_actor(blueprint, spawn_transform)
        self.walker2.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), self.ped_speed))
        self.walker2.icr = ICR.VERY_LOW
        self.walker2.son = SON.YIELDING
        self.walker2.initial_son = SON.YIELDING
        # Walker2: Walk along the sidewalk
        self.path_w2 = self._compute_plans(
            [(0, 2), (0, 4), (0, 8), (0, 20)],
            spawn_transform.location,
            color=carla.Color(r=0, g=0, b=255) if self.debug else None,
        )
        self.path_controller_w2 = PathController(self.world, self.walker2, self.path_w2, self.ped_speed)

        # self.w2_look_right = TurnHeadRightWalk(walker=self.walker2, start_pos=self.path_w2[1])
        self.w2_raise_arm = Greet(
            walker=self.walker2, start_pos=self.path_w2[1], char="yielding", end_pos=self.path_w2[2]
        )

    def setup(self, obstacles, conf):
        self.setup_w1(obstacles[0][0], obstacles[0][1], conf)
        self.setup_w2(obstacles[1][0], obstacles[1][1], conf)

        self.world.tick()

        self.update_db()

    def restart_scenario_specific(self, scenario, conf=ControllerConfig()):
        self.choice = None
        obstacles = scenario[1]
        self.setup(obstacles, conf)
        spawn_point = self.spawn_point
        cam_transform = carla.Transform(
            carla.Location(
                spawn_point.location.x - 20, spawn_point.location.y - conf.spawning_distance, spawn_point.location.z + 4
            ),
            carla.Rotation(-20, -40, 0),
        )
        self.world.get_spectator().set_transform(cam_transform)
        # p = self.player.get_location()
        # print(p)
        # self.world.debug.draw_point(p+carla.Location(0,-2,2), size=0.1, color=carla.Color(r=0,g=255,b=255), life_time=0)
        if not self.random:
            self.player.set_target_velocity(carla.Vector3D(0, -conf.car_speed, 0))
