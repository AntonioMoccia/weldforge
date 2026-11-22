# A recognised weld seam.
#
# Lengths are in METRES and angles in RADIANS: this is a ROS interface, so it
# follows REP-103 even though the planning core works in millimetres. The
# conversion happens once, in weldforge_cell.conversions.
std_msgs/Header header

string id
string label
# fillet | butt | lap | corner | edge | tee | plug | unknown
string joint_type

# One pose per station, Z along the wire and X along the travel direction, so
# a controller can use the pose directly as a tool target.
geometry_msgs/Pose[] stations

float64 length                  # arc length along the seam, m
float64 groove_angle            # included angle on the weld side, rad
float64 groove_angle_spread     # peak-to-peak variation along the seam, rad
# EN ISO 6947: PA PB PC PD PE PF PG
string welding_position
float64 thickness_a
float64 thickness_b
float64 root_gap
float64 confidence              # 0..1 from the classifier
bool closed
bool selected
bool spans_two_solids
string solid_a_id
string solid_b_id
string[] notes
