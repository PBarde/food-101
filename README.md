## Install environment (I use conda on mila-cluster)

Should be as simple as running 
```
conda env create -f freeze.yml
```
otherwise you will also find the files from if you prefer using these:
```
pip freeze > pip_freeze.txt
conda list -e > conda_list.txt
```

Finally, if it still does not work, I used the following command lines 
```
conda create -y -n take_home python=3.10 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -c conda-forge
conda activate take_home 
conda install argparse
conda install pathlib 
conda install numpy
conda install tqdm
conda install json
conda install matplotlib
conda install seaborn
conda install pandas
conda install pip
pip install torchsummary
```


## Getting the data
the code expects the data to be under `/tmp/food-101/`
copy data to `/tmp`
```
rsync -rzpv /home/mila/b/bardepau/scratch/food-101/food-101 /tmp/
```

## Folder structure
```
food-101/
|
|-- process_data.py
|-- train.py
|-- plot_train.py
|-- evaluate.py
|-- plot_evaluate.py
|-- utils.py
| ... 
|-- my_submit_run/
...     | ...
        | -- best_model.pth 
        | ...
```

## Note:
all the scripts can be queried with
```
python a_script.py --help 
```

## Required steps:
get the data in the proper folder, please run: 
```
python process_data.py
```

to evaluate submitted model
```
python evaluate.py --save_path my_submit_run --eval_data_path /path/to/test/folder
```

note that  `/path/to/test/folder` is expected to be structured as follows
```
root/dog/xxx.png
root/dog/xxy.png
root/dog/[...]/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/[...]/asd932_.png
```

## Optional
get a look at the model we are going to use:
```
python model.py
```

run a training for 3 epochs
```
python train.py --n_epochs 3 --save_path run_test
```
make training plots for this training
```
python plot_train.py --save_path run_test
```

evaluate this training
```
python plot_train.py --save_path run_test 
```
plot evaluation results for this training
```
python plot_evaluate.py --save_path run_test
```
plot evaluation results for submitted run
```
python plot_evaluate.py --save_path my_submit_run
```
