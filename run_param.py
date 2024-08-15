from MAxPy import maxpy
from MAxPy import probprun
from testbench import testbench_run

circuit = maxpy.AxCircuit(top_name="conv_acc")
circuit.set_testbench_script(testbench_run)

circuit.set_group("study_complete_4")
circuit.set_synth_tool(None)
circuit.set_results_filename("output.csv")
circuit.parameters = {
    "[[MULTIPLIER_TYPE]]": ["DRUMs"],
    "[[MULTIPLIER_K]]": ["15", "14", "13", "12", "11", "10", "9", "8", "7", "6", "5", "4"],
    "[[ADDER_TYPE]]": ["copyA", "copyB", "eta1", "loa", "trunc0" ],
    "[[ADDER_K]]": ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"]
}
circuit.rtl2py_param_loop(base="rtl_param")

# generate pareto front
pareto_area = circuit.get_pareto_front_reduction_mape_outliers("area", "mape", maxX=1, maxY=0)
pareto_power = circuit.get_pareto_front_reduction_mape_outliers("power", "mape", maxX=1, maxY=0)
pareto_timing = circuit.get_pareto_front_reduction_mape_outliers("timing", "mape", maxX=1, maxY=0)

pareto_area = circuit.get_pareto_front("area", "mae", maxX=0, maxY=0)
pareto_power = circuit.get_pareto_front("power", "mae", maxX=0, maxY=0)
pareto_timing = circuit.get_pareto_front("timing", "mae", maxX=0, maxY=0)

circuit.set_results_filename("output_probrun.csv")
probprun.probprun_loop(circuit, pareto_power)
