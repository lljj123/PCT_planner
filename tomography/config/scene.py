class ScenePCD():
    file_name = None
    # [roll_x, pitch_y, yaw_z] in degrees.
    # Use x/y to correct a tilted point cloud so world z becomes vertical again.
    rot_deg = [0.0, 0.0, 0.0]
    auto_align_ground = False
    ground_seed_percentile = 35.0
    ground_ransac_dist = 0.08
    ground_ransac_n = 3
    ground_ransac_iters = 1000


class SceneMap():
    resolution = 0.10
    ground_h = 0.0
    slice_dh = 0.5


class SceneTrav():
    kernel_size = 7
    interval_min = 0.50
    interval_free = 0.65
    slope_max = 0.36
    step_max = 0.20
    standable_ratio = 0.20
    cost_barrier = 50.0

    safe_margin = 0.4
    safe_margin_gamma=1
    inflation = 0.2
