python baseline.py --dataset german --sensitive age --validation_size 0.05 --test_size 0.05 --method random --budget 50 --n_run 5
python baseline.py --dataset german --sensitive sex --validation_size 0.05 --test_size 0.05 --method random --budget 50 --n_run 5
python baseline.py --dataset compas --sensitive race --validation_size 0.05 --test_size 0.05 --method random --budget 50 --n_run 5
python baseline.py --dataset compas --sensitive sex --validation_size 0.05 --test_size 0.05 --method random --budget 50 --n_run 5
python baseline.py --dataset bank --sensitive age --validation_size 0.05 --test_size 0.05 --method random --budget 50 --n_run 5
python baseline.py --dataset bank --sensitive marital --validation_size 0.05 --test_size 0.05 --method random --budget 50 --n_run 5
python baseline.py --dataset adult --sensitive sex --validation_size 0.05 --test_size 0.05 --method random --budget 50 --n_run 5
python baseline.py --dataset adult --sensitive race --validation_size 0.05 --test_size 0.05 --method random --budget 50 --n_run 5