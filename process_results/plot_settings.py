import matplotlib
matplotlib.use('pgf')
import numpy as np

def hash(text:str):
  hash=0
  for ch in text:
    hash = ( hash*281 ^ ord(ch)*997) & 0xFFFFFFFF 
  return hash # returns an integer number between 0 and 2**32

def format(post_gen_selection_method:str, feedback_method:str):
    rng = np.random.default_rng(hash(post_gen_selection_method + feedback_method)+1)
    color =         {'last': 'tab:gray',
                    'mbr_hard': 'tab:blue',
                    'mbr_soft': 'tab:orange',
                    'er_hard': 'tab:red',
                    'er_soft': 'tab:green',
                    'codet_hard': 'tab:cyan',
                    'codet_soft': 'tab:pink',
                    'Oracle-Test Accuracy with Oracle-Test Feedback': 'm',
                    'Oracle-Test Accuracy with Self-Test Feedback': 'k',
                    'Self-Test Accuracy with Self-Test Feedback': 'c',
                    'QiskitHumanEvalSim': 'k',
                    'QiskitHumanEvalSimX': 'r',
                    'Qwen3 (235B-A22B-Instruct)': '#1D1C1A',
                    'Qwen3-Next (80B-A3B-Instruct)': "#2E9CC5",
                    'Qwen3-Coder-Next (80B-A3B-Instruct)': "#2DAB2D",
                    }
    random_color = rng.choice(matplotlib.pyplot.rcParams['axes.prop_cycle'].by_key()['color'])
    line_style =    {'codes_on_tests': '-',
                    'thompson_sampling_tf': '--',
                    'thompson_sampling_cf': ':',
                    'thompson_sampling_cfs': '-.',
                    'last': '--',
                    'mbr_hard': '-',
                    'mbr_soft': ':',
                    'er_hard': '-',
                    'er_soft': '-.',
                    'codet_hard': '--',
                    'codet_soft': ':',
                    'Oracle-Test Accuracy with Oracle-Test Feedback': ':',
                    'Oracle-Test Accuracy with Self-Test Feedback': '-',
                    'Self-Test Accuracy with Self-Test Feedback': '--',
                    'QiskitHumanEvalSim': '-',
                    'QiskitHumanEvalSimX': ':',
                    'Qwen3 (235B-A22B-Instruct)': '-',
                    'Qwen3-Next (80B-A3B-Instruct)': "--",
                    'Qwen3-Coder-Next (80B-A3B-Instruct)': ":",
                    }
    random_line_style = rng.choice(['-', '--', '-.', ':'])

    return {'color': color.get(post_gen_selection_method, random_color), 'linestyle': line_style.get(feedback_method, random_line_style)}
