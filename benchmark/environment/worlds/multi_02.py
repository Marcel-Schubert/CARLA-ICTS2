from benchmark.environment.world import World, ControllerConfig

from benchmark.environment.car_controller import CarController
from benchmark.environment.ped_controller import *

import carla


class WorldMulti02(World):
    def __init__(self, carla_world, hud, scenario, args):
        super().__init__(carla_world, hud, scenario, args)
        self.debug = True

    def tick_w1(self):
        w1_approaching = self.w1_path_controller_approach.step() != "Done"
        self.w1_look_right.step()
        if w1_approaching:
            return

        if self.w1_choice is None:
            distance = y_distance(self.walker.get_location(), self.player.get_location()) - 2
            if self.decision_trigger(distance, self.db, without_speed=True):
                self.w1_choice = "Wait"
                self.walker.icr = ICR.PLANNING_TO
                self.walker.son = SON.YIELDING
            else:
                self.w1_choice = "Cross"
                self.walker.icr = ICR.GOING_TO
                self.walker.son = SON.FORCING
        elif self.w1_choice == "Wait" or self.w2_choice != "Cross":
            self.w1_wait.step()
            # Note: db is updated each tick
            distance = y_distance(self.walker.get_location(), self.player.get_location()) - 2
            if not self.decision_trigger(distance, self.db):
                self.w1_choice = "Cross"
        elif self.w1_choice == "Cross" and self.w2_choice == "Cross":
            self.w1_iss_cross.step()
            self.w1_path_controller_cross.step()
        if not self.w1_path_controller_cross.done:
            return

        self.w1_iss_crossed.step()
        self.w1_path_controller_continue.step()

    def tick_w2(self):
        w2_approaching = self.w2_path_controller_approach.step() != "Done"
        self.w2_look_right.step()
        if w2_approaching:
            return

        if self.w2_choice is None:
            distance = y_distance(self.walker2.get_location(), self.player.get_location()) - 2
            if self.decision_trigger(distance, self.db, without_speed=True):
                self.w2_choice = "Wait"
                self.walker2.icr = ICR.PLANNING_TO
                self.walker2.son = SON.YIELDING
            else:
                self.w2_choice = "Cross"
                self.walker2.icr = ICR.GOING_TO
                self.walker2.son = SON.FORCING
        elif self.w2_choice == "Wait" or self.w1_choice != "Cross":
            self.w2_wait.step()
            # Note: db is updated each tick
            distance = y_distance(self.walker2.get_location(), self.player.get_location()) - 2
            if not self.decision_trigger(distance, self.db):
                self.w2_choice = "Cross"
        elif self.w2_choice == "Cross" and self.w1_choice == "Cross":
            self.w2_iss_cross.step()
            self.w2_path_controller_cross.step()

        if not self.w2_path_controller_cross.done:
            return

        self.w2_iss_crossed.step()
        self.w2_path_controller_continue.step()

    def tick(self, clock):
        super().tick(clock)
        self.update_db()
        self.tick_w1()
        self.tick_w2()

    def update_db(self):
        crossing_width = 11  # 11 m
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
        self.w1_path_approach = self._compute_plans(
            offsets=[(0, 10), (0.5, 11), (1, 11)],
            position=spawn_transform.location,
            color=carla.Color(r=255, g=0, b=0) if self.debug else None,
        )
        self.w1_path_controller_approach = PathController(
            self.world, self.walker, self.w1_path_approach, self.ped_speed
        )

        self.w1_look_right = TurnHeadRightWalk(
            walker=self.walker, start_pos=self.w1_path_approach[0], end_pos=self.w1_path_approach[-1]
        )

        # Walk to the other side of the road
        self.w1_iss_cross = InternalStateSetter(self.walker, start_pos=None, icr=ICR.GOING_TO, son=SON.FORCING)
        self.w1_path_cross = self._compute_plans(
            [(11, 11)], spawn_transform.location, color=carla.Color(r=0, g=255, b=0) if self.debug else None
        )
        self.w1_path_controller_cross = PathController(
            self.world, self.walker, self.w1_path_cross, self.ped_crossing_speed
        )

        self.w1_path_continue = self._compute_plans(
            [(12.5, 12), (12.5, 20)],
            spawn_transform.location,
            color=carla.Color(r=0, g=255, b=0) if self.debug else None,
        )
        self.w1_path_controller_continue = PathController(
            self.world, self.walker, self.w1_path_continue, self.ped_speed
        )

        self.w1_wait = PathController(self.world, self.walker, [], 0)

        self.w1_iss_crossed = InternalStateSetter(
            self.walker, self.w1_path_cross[-1], icr=ICR.VERY_LOW, son=SON.AVERTING
        )

    def setup_w2(self, blueprint, spawn_transform: carla.Transform, conf):
        spawn_transform.location.y -= conf.spawning_distance
        self.walker2 = self.world.try_spawn_actor(blueprint, spawn_transform)
        self.walker2.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), self.ped_speed))
        self.walker2.icr = ICR.LOW
        self.walker2.son = SON.YIELDING
        self.walker2.initial_son = SON.YIELDING if conf.char == "yielding" else SON.FORCING
        # Walker2: Walk along the sidewalk
        self.w2_path_approach = self._compute_plans(
            [(0, 10), (2, 12.5), (2.5, 12.5)],
            spawn_transform.location,
            color=carla.Color(r=0, g=0, b=255) if self.debug else None,
        )
        self.w2_path_controller_approach = PathController(
            self.world, self.walker2, self.w2_path_approach, self.ped_speed
        )
        self.w2_look_right = TurnHeadRightWalk(
            walker=self.walker2, start_pos=self.w2_path_approach[0], end_pos=self.w2_path_approach[1]
        )

        self.w2_wait = PathController(self.world, self.walker2, [], 0)

        self.w2_iss_cross = InternalStateSetter(self.walker2, start_pos=None, icr=ICR.GOING_TO, son=SON.FORCING)

        self.w2_path_cross = self._compute_plans(
            [(12.5, 12.5)],
            spawn_transform.location,
            color=carla.Color(r=0, g=0, b=255) if self.debug else None,
        )

        self.w2_path_controller_cross = PathController(
            self.world, self.walker2, self.w2_path_cross, self.ped_crossing_speed
        )

        self.w2_iss_crossed = InternalStateSetter(
            self.walker2, self.w2_path_cross[-1], icr=ICR.VERY_LOW, son=SON.AVERTING
        )

        self.w2_path_continue = self._compute_plans(
            [(12.5, 15.2), (12.5, 20)],
            spawn_transform.location,
            color=carla.Color(r=0, g=0, b=255) if self.debug else None,
        )

        self.w2_path_controller_continue = PathController(
            self.world,
            self.walker2,
            self.w2_path_continue,
            self.ped_speed / 2,
            speed_schedule=[(self.w2_path_continue[0], 2)],
        )

    def setup(self, obstacles, conf):
        self.setup_w1(obstacles[0][0], obstacles[0][1], conf)
        self.setup_w2(obstacles[1][0], obstacles[1][1], conf)

        self.world.tick()

        self.update_db()

    def restart_scenario_specific(self, scenario, conf=ControllerConfig()):
        self.w1_choice = None
        self.w2_choice = None
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
