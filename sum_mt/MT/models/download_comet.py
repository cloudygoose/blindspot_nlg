import torch
from comet import download_model, load_from_checkpoint

model_path1 = download_model("wmt20-comet-da")

model1 = load_from_checkpoint(model_path1)

torch.save(model1, "wmt20-comet-da.save")
print('saving wmt20-comet-da.save')

model_path2 = download_model('wmt21-comet-qe-mqm')

model2 = load_from_checkpoint(model_path2) 

torch.save(model2, 'wmt21-comet-qe-mqm.save')
print('saving wmt21-comet-qe-mqm.save')  

