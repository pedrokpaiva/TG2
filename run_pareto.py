from MAxPy import maxpy
from MAxPy import probprun
from testbench import testbench_run

circuit = maxpy.AxCircuit(top_name="poly1")
circuit.set_testbench_script(testbench_run)

circuit.set_group("study_no_1")
circuit.set_synth_tool("yosis")
circuit.set_results_filename("output_probrun.csv")

pareto_power = circuit.get_pareto_front("power", "mape", maxX=0, maxY=0)

# circuit.set_results_filename("output_probrun.csv")
# probprun.probprun_loop(circuit, pareto_power)
