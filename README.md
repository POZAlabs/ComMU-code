# ComMU (A Dataset for Combinatorial Music Generation)

- [Paper on arXiv](https://paper_page)
- [Demo Page](https://pozalabs.github.io/ComMU/)
- [Dataset](https://github.com/POZAlabs/ComMU-code/tree/master/dataset)

## Getting Started
### Setup
1. Clone this repository
2. Install required packages
```
pip install -r requirements.txt
```
### Download the Data
1. download csv with meta information outside the root directory
- csv file consists of meta information of each midi file
- min/max velocity value is not contained for being calculated from the raw midi file
2. download midifiles in subdirectory named 'raw'
```
.
├── commu_meta.csv
└── root
    └── raw
        └── midifiles(.mid)
```
3. After successful preprocessing, project tree would like be this,
```
.
├── commu_meta.csv
└── root
    ├── train
    │   ├── raw
    │   ├── augmented_tmp
    │   ├── augmented
    │   └── npy_tmp
    ├── val
    │   ├── raw
    │   ├── augmented_tmp
    │   ├── augmented
    │   └── npy_tmp
    └── output_npy
        ├── input_train.npy
        ├── input_val.npy
        ├── target_train.npy
        └── target_val.npy
```


## Preprocessing
```
python3 preprocess.py --root_dir /root --csv_path /commu_meta.csv
```

## Training
```
python3 -m torch.distributed.launch --nproc_per_node=4 ./train.py --data_dir /root/output_npy --work_dir /working_direcoty
```

## Generating
```
python3 generate.py \
--checkpoint_dir /working_direcoty/checkpoint_best.pt \
--output_dir /output_dir
--bpm 70 \
--audio_key aminor \
--time_signature 4/4 \
--pitch_range mid_high \
--num_measures 8 \
--inst acoustic_piano \
--genre newage \
--min_velocity 60 \
--max_velocity 80 \
--track_role main_melody \
--rhythm standard \
--chord_progression Am-Am-Am-Am-Am-Am-Am-Am-G-G-G-G-G-G-G-G-F-F-F-F-F-F-F-F-E-E-E-E-E-E-E-E-Am-Am-Am-Am-Am-Am-Am-Am-G-G-G-G-G-G-G-G-F-F-F-F-F-F-F-F-E-E-E-E-E-E-E-E \
--num_generate 3
```
