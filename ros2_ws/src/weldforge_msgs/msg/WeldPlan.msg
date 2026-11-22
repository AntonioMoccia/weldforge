# A complete plan: what to weld, how, and where the part sits.
std_msgs/Header header
string job_id
string job_name
string part_name
string cell_name
string robot_name

weldforge_msgs/WeldSeam[] seams
weldforge_msgs/WeldProcedure[] procedures
# Part origin in the cell frame, as chosen by the placement search.
geometry_msgs/Pose part_pose
bool placement_feasible
float64 placement_score
string[] placement_violations

# One trajectory per pass, in plan order. Indices into `procedures` and into
# each procedure's `passes` travel alongside so the welder node knows which
# setpoints belong to which trajectory.
trajectory_msgs/JointTrajectory[] trajectories
string[] trajectory_seam_ids
uint16[] trajectory_pass_indices
# Per trajectory point: true where the arc must be burning. Flattened in the
# same order as the trajectories, with `arc_on_offsets` giving each
# trajectory's start index -- ROS messages cannot nest variable-length arrays.
bool[] arc_on
uint32[] arc_on_offsets

weldforge_msgs/JobReport report
