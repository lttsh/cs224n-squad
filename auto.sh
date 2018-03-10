python code/main.py --experiment_name=bidaf_GRU --model_name=bidaf --rnn_cell=GRU \
--num_epochs=12 --mode=train

python code/main.py --experiment_name=bidaf_LSTM --model_name=bidaf --rnn_cell=LSTM \
--num_epochs=12 --mode=train

python code/main.py --experiment_name=bidaf_GRU_dp_0.3 --model_name=bidaf --rnn_cell=GRU \
--num_epochs=5 --mode=train --dropout=0.3

python code/main.py --experiment_name=bidaf_LSTM_dp_0.3 --model_name=bidaf --rnn_cell=LSTM \
--num_epochs=5 --mode=train --dropout=0.3


