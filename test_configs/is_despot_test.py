class TestConfig:
    PI = 3.14159

    simulation_step = 0.05  # 0.008
    sensor_simulation_step = '0.5'
    synchronous = True
    segcam_fov = '90'
    segcam_image_x = '400'  # '1280'
    segcam_image_y = '400'  # '720'

    grid_size = 2  # grid size in meters
    speed_limit = 50
    max_steering_angle = 1.22173  # 70 degrees in radians
    occupancy_grid_width = '1920'
    occupancy_grid_height = '1080'
    display = False
    location_threshold = 1.0



    ped_speed_range = [2.0, 2.0]
    ped_distance_range = [25, 25]
    # car_speed_range = [6, 9]
    scenarios = ['01_int','02_int', '03_int', "04_int", "05_int", '01_non_int', '02_non_int', '03_non_int']#,"02_int", "03_int"  #, '02', '03', '04', '05', '06', '07', '08', '09']

    # Simulator Parameters
    host = '127.0.0.1'
    port = 2000
    width = 1920
    height = 1080
    display = False
    filter = 'vehicle.audi.tt'
    rolename = 'hero'
    gama = 1.7
    N_DISCRETE_ACTIONS = 3
    max_speed = 30 * 0.27778  # in m/s #TODO change max speed
    hit_penalty = 1000
    goal_reward = 1000
    braking_penalty = 1
    record = True


    # ISDESPOT parameters
    num_steps = 500
    despot_port = 1245

