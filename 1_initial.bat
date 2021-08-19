python initial_model.py --dataset german --sensitive age --validation_size 0.05 --test_size 0.05 --n_run 5 --gpu -1
python initial_model.py --dataset german --sensitive sex --validation_size 0.05 --test_size 0.05 --n_run 5 --gpu -1
python initial_model.py --dataset compas --sensitive race --validation_size 0.05 --test_size 0.05 --n_run 5 --gpu -1
python initial_model.py --dataset compas --sensitive sex --validation_size 0.05 --test_size 0.05 --n_run 5 --gpu -1
python initial_model.py --dataset bank --sensitive age --validation_size 0.05 --test_size 0.05 --n_run 5 --gpu -1
python initial_model.py --dataset bank --sensitive marital --validation_size 0.05 --test_size 0.05 --n_run 5 --gpu -1
python initial_model.py --dataset adult --sensitive sex --validation_size 0.05 --test_size 0.05 --n_run 5 --gpu -1
python initial_model.py --dataset adult --sensitive race --validation_size 0.05 --test_size 0.05 --n_run 5 --gpu -1