from MAxPy import maxpy
from testbench import testbench_run
circuit = maxpy.AxCircuit(top_name="conv_acc")
circuit.set_testbench_script(testbench_run)
circuit.set_results_filename("output.csv")

# basic flow RTL level
circuit.set_synth_tool(None)
circuit.rtl2py(target="exact", run_tb=1)

# # basic flow gate level
# circuit.set_synth_tool("yosys")
# circuit.rtl2py(target="exact_yosys", run_tb=0)
