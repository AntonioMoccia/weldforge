# A welding procedure specification bound to one seam.
string wps_id
string seam_id
string base_material
string filler
string standard
string welding_position
string joint_type
float64 thickness               # governing member thickness, m
float64 target_leg              # m
float64 preheat                 # degrees C
float64 interpass_max           # degrees C
float64 heat_input_max          # J/m, 0 = unconstrained
weldforge_msgs/WeldPass[] passes
string[] warnings
