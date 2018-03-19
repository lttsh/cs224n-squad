EXPERIMENT1=$1
EXPERIMENT2=$2
cl work main::cs224n-omelette-du-fromage
echo "Uploading code"
cl upload code

echo "Uploading checkpoint"
cl upload experiments/$EXPERIMENT1/best_checkpoint -n checkpoint1
cl upload experiments/$EXPERIMENT2/best_checkpoint -n checkpoint2

echo "Creating predictions"
cl run --request-memory 10g --request-disk 2g --name gen-answers --request-docker-image abisee/cs224n-dfp:v4 \
 :code :checkpoint1 :checkpoint2 glove.txt:0x97c870/glove.6B.100d.txt data.json:0x4870af \
 'python code/main.py --question_len=20 --hidden_size=150 --selfattn_size=100 --model_name=stack --mode=ensemble_write --num_layers=1 \
 --glove_path=glove.txt --json_in_path=data.json --ckpt_load_dir=checkpoint1 && \
 python code/main.py --question_len=20 --hidden_size=100 --selfattn_size=64 --model_name=pointer --mode=ensemble_write --num_layers=1 \
 --glove_path=glove.txt --json_in_path=data.json --ckpt_load_dir=checkpoint2 &&
python code/main.py --json_in_path=data.json --glove_path=glove.txt --experiment_name=ensemble --mode=ensemble_predict'

cl wait --tail gen-answers

echo "Evaluating predictions\n"
cl run --name run-eval --request-docker-image abisee/cs224n-dfp:v4 \
:code data.json:0x4870af preds.json:gen-answers/predictions.json \
'
python code/evaluate.py data.json preds.json
'

cl wait --tail run-eval
