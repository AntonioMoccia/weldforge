# One pass of one seam: the numbers that go on the power source and the robot.
uint16 index
string name                     # root | fill n | cap | single

# Deposit geometry, metres.
float64 leg
float64 throat
float64 offset_across
float64 offset_normal

# Process. ISO 4063 reference number in iso_4063.
string process                  # GMAW | FCAW | GTAW | SMAW | SAW | LASER
uint16 iso_4063
string transfer_mode            # short_arc | spray | pulse | double_pulse | cmt | ...
float64 current                 # A
float64 voltage                 # V
float64 wire_speed              # m/s  (shop sheets quote m/min; converted here)
float64 wire_diameter           # m
string gas                      # EN ISO 14175 designation
float64 gas_flow                # l/min -- a volumetric flow, left as supplied
string polarity

# Motion.
float64 travel_speed            # m/s
float64 work_angle              # rad, from the surface of flank A
float64 travel_angle            # rad, negative = push
float64 stickout                # m, contact tip to work

# Weaving. Executed by the controller, not interpolated into the path.
string weave_shape              # none | zigzag | triangular | circular | trochoidal
float64 weave_amplitude         # m, peak to peak
float64 weave_frequency         # Hz
float64 weave_dwell             # s at each side

# Timing.
float64 arc_start_delay
float64 crater_fill
float64 burnback

float64 heat_input              # J/m
