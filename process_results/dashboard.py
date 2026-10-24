import gradio as gr
import numpy as np
import pandas as pd
from utils import read_jsonl  # assumes read_jsonl(path) returns a list of dicts with 'code', 'tests', 'scores'

def retrieve_data(result_path, max_iteration_round):
    result = read_jsonl(result_path)
    if not result:
        return "No data found.", pd.DataFrame()
    
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
    result_path_input = gr.Textbox(label="Result File Path")
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

demo.launch(share=True)