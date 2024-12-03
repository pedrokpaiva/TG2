from MAxPy import maxpy
from testbench_cnn_debug import testbench_run
circuit = maxpy.AxCircuit(top_name="conv_acc")
circuit.set_testbench_script(testbench_run)
circuit.set_results_filename("output.csv")

circuit.set_synth_tool(None)
circuit.pymod_path = "conv_acc_exact"
circuit.run_testbench()

