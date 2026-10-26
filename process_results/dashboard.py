import json
import os
import gradio as gr
import numpy as np
import pandas as pd

# Only result files under the repository's results/ directory may be loaded
RESULTS_DIR = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))

def retrieve_data(result_path, max_iteration_round):
    # The path comes from the UI, so it is validated and read here rather than passed to read_jsonl
    full_path = os.path.realpath(os.path.join(RESULTS_DIR, result_path))
    if not full_path.startswith(RESULTS_DIR + os.sep):
        return "Invalid path: result files must be inside results/.", "", pd.DataFrame()
    if not os.path.isfile(full_path):
        return "No data found.", "", pd.DataFrame()
    with open(full_path, 'r') as file:
        result = [json.loads(line) for line in file if line.strip()]
    if not result:
        return "No data found.", "", pd.DataFrame()
    
    codes = [row['code'] for row in result[:max_iteration_round]]
    scores_matrix = [code['scores'][:max_iteration_round] for code in codes]
    scores_matrix = [[round(test_score*100, 0) for test_score in code_scores] for code_scores in scores_matrix]
    observation_history = [observation['insight'] for observation in result[max_iteration_round-1]['observation_history']]

    # Compute best code
    aggregates = np.mean(np.array(scores_matrix), axis=1)
    best_idx = np.argmax(aggregates)
    best_code = codes[best_idx]['code']

    df = pd.DataFrame(scores_matrix, 
                      [f"C{i+1}" for i in range(max_iteration_round)], 
                      [f"T{i+1}" for i in range(max_iteration_round)])

    df.insert(0, r"Code\Test", [f"C{i+1}" for i in range(max_iteration_round)])
    df["Aggregate"] = np.round(aggregates)

    return f"Code {best_idx+1} is most promising:\n```python\n{best_code}\n```", "-"+"\n-".join(observation_history), df

with gr.Blocks() as demo:
    gr.Markdown("# Bayesian Code Assistant")
    result_path_input = gr.Textbox(label="Result File Path (relative to results/)")
    round_slider = gr.Slider(label="Maximum Iteration Round", minimum=1, maximum=25, step=1, value=1)
    run_button = gr.Button("Load & Evaluate")
    table_output = gr.Dataframe(label="Code vs Test Suite Matrix", column_widths = ["4%"]*25, )
    observations_output = gr.Textbox(label="Observations", interactive=False, lines=25)
    summary_output = gr.Textbox(label="Summary", interactive=False, lines=25)

    run_button.click(
        retrieve_data,
        inputs=[result_path_input, round_slider],
        outputs=[summary_output, observations_output, table_output]
    )

demo.launch()