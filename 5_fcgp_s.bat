python fcgp_s.py --dataset german --sensitive age --validation_size 0.05 --test_size 0.05 --balance 0.5 --optimize grid --budget 50 --n_run 5
python fcgp_s.py --dataset german --sensitive sex --validation_size 0.05 --test_size 0.05 --balance 0.5 --optimize grid --budget 50 --n_run 5
python fcgp_s.py --dataset compas --sensitive race --validation_size 0.05 --test_size 0.05 --balance 0.5 --optimize grid --budget 50 --n_run 5
python fcgp_s.py --dataset compas --sensitive sex --validation_size 0.05 --test_size 0.05 --balance 0.5 --optimize grid --budget 50 --n_run 5
python fcgp_s.py --dataset bank --sensitive age --validation_size 0.05 --test_size 0.05 --balance 0.5 --optimize grid --budget 50 --n_run 5
python fcgp_s.py --dataset bank --sensitive marital --validation_size 0.05 --test_size 0.05 --balance 0.5 --optimize grid --budget 50 --n_run 5
python fcgp_s.py --dataset adult --sensitive sex --validation_size 0.05 --test_size 0.05 --balance 0.5 --optimize grid --budget 50 --n_run 5
python fcgp_s.py --dataset adult --sensitive race --validation_size 0.05 --test_size 0.05 --balance 0.5 --optimize grid --budget 50 --n_run 5