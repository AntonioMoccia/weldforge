# What gets written to the power source for one pass.
#
# On a synergic machine the job number and wire speed command the arc, and the
# current and voltage are what the characteristic should produce -- they are
# carried for the record and for monitoring, not as commands.
uint16 pass_index
bool has_job_number
int32 job_number
float64 current
float64 voltage
float64 wire_speed              # m/s
float64 arc_length_correction   # -10..+10 on most brands
float64 dynamic_correction
float64 gas_flow                # l/min
float64 pulse_frequency         # Hz
string machine_profile

# Raw fieldbus image, as produced by the machine profile.
string[] analog_names
int32[] analog_values
string[] digital_names
bool[] digital_values
string[] notes
