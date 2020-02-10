#!/bin/bash
python main.py --mode=max --model_idx=2 --batch_size=16 --flatten && 
python main.py --mode=min --model_idx=2 --batch_size=16 --flatten &&
python main.py --mode=max --model_idx=4 --batch_size=5 --single_channel