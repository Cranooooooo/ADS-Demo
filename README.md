# Evolve Path Tracer: Early Detection of Malicious Addresses in Cryptocurrency
The demo codes for:
+ D1: Result Reproduce on Three Datasets
+ D2: Data Preparation for User Specified Address
+ D3: Prediction for User Specified Address

## Requirements
Please download pre-process-data and blocksci-made-data from:
https://drive.google.com/drive/folders/1PB5O9VfsajaOfD5vlhBSwAvprKRQuXUF?usp=sharing

To build environments:
`conda activate env-x`;
`pip3 install -r requirement_env_x.txt`

+ For D1 and D3, please create an environment with `requirement_env_1.txt` .
+ For D2, please create an environment with `requirement_env_2.txt`.


## Usage
### Folder and File Description
+ `ckpt/`: folder saves the trained models (checkpoints).
+ `misc/`: folder contains dataset information and loss wrapper model.
+ `model/`: folder contains Evolve Path Tracer and related modules.
+ `path_related_tool_box/`: folder contains codes for data preparation.
+ `pre_process_data/`: folder contains training and testing data. (Decompress `pre_process_data.tar.gz` to get the folder)
+ `blocksci_made_data_bk_up_2/`: folder contains Blocksci data we pre-processed. (Decompress `blocksci_made_data.tar.gz` to get the folder)
+ `demo_performance_notebook.ipynb`: D1: result reproduce on three datasets.
+ `demo_path_prepare.ipynb`: D2: data preparation for user specified address.
+ `demo_user_specified_address_predict.ipynb`: D3: prediction for user specified address.
+ `opts.py`: configuration file.
+ `dataloader.py`: dataloader.
+ `train.py`: continue training base on our pretrained models.

Note: 
#### (1) For Blocksci, if prepare data from scratch, you will need ~1.5TB storage and 64GB RAM.
#### (2) The default case is a Hacker BTC address(`bc1q93ecep2338dy9aauwyvh4g22t49rnedxl8z0tj`) on the 2017 Binance hack incident.

### D1: result reproduce on three datasets (Please use env-1)
In this notebook, the code will first load three models on 'Hack', 'Ransomware', and 'Darknet' datasets.
Then we evaluate their performance respectively.
The warning signal is due to the incompatibility between jupyter and our environment.


### D2: Data Preparation for User Specified Address (Please use env-2)
In this notebook, users can specify the address they interest in by changing the hash_str in the second block.
The code will prepare the necessary data (first 24 hours) (Address Feature, FR/BK Asset Transfer Path, FR/BK Asset Transfer Path Graph) for the given address .
The data will be saved in `tmp_Path_Graph_data/`.
We also add a block for path visualization for better interpretation of our asset transfer paths and graphs.
(Note, we merge the source and destination node for visual clarity)

### D3: Prediction for User Specified Address (Please use env-1)
In this block, user can predict the address label with the data prepared from the D2 demo.
The code will preprocess the data first.
Then we use the data and predict the label with three models respectively.
We also add a block for Benevolent Rate (B.R.) evolution visualization for better interpretation of each kind of feature's contribution.
