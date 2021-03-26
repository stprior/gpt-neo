 
import json
from pprint import pprint
path_to_local_weights="/home/stephen/the-eye.eu/public/AI/gptneo-release/GPT3_2-7B"
pretrained_model="GPT3_2-7B"
path_to_model = "gs://pt-gpt3-27b3/" #@param {type:"string"}
batch_size = 8 #@param {type:"integer"}
dset = ""  #@param {type:"string"}
mesh_shape = "x:4,y:2" #@param {type:"string"}
train_steps = 1000 #@param {type:"integer"}
steps_per_checkpoint = 500 #@param {type:"integer"}
start_step = 400000 if pretrained_model == "GPT3_2-7B" else 362000
dataset = "Sampling_Only"

if path_to_model == "":
  path_to_model = f'{bucket_base.strip("/")}/{pretrained_model}'
  print(f'MODEL PATH: {path_to_model}\n')

if dset == "" and dataset != "Sampling_Only":
  dset = dataset
elif dataset is None:
  dset = "pile"

def pad_to_multiple_of(n, mult):
    """
    pads n to a multiple of mult
    """
    extra = n % mult
    if extra > 0:
        n = n + mult - extra
    return n

with open(f'{path_to_local_weights}/config.json', 'r') as f:
    data = json.load(f)
    pprint(data)
    dset_val = [[dset, None, None, None]] if dset != "" else data["datasets"]
    mods = {
            "mesh_shape": mesh_shape,
            "layout": "intermediate_expanded:x,heads:x,memory_length:y,embd:y",
            "model_path": path_to_model,
            "datasets": dset_val,
            "train_steps": start_step + train_steps,
            "eval_steps": 0,
            "train_batch_size": batch_size,
            "predict_batch_size": batch_size
          }
    data.update(mods)
    print('\n--->\n')
    pprint(data)
    with open(f'configs/{pretrained_model}.json', 'w') as outfile:
      json.dump(data, outfile, indent=2)
