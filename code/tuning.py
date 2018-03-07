import os

NUM_EPOCHS=10
learning_rate={0.1, 0.05, 0.01, 0.005}
max_grad_norm={5.0}
dropout={0.0, 0.15, 0.20, 0.30}
hidden_size={100, 200}

for lr in learning_rate:
  for mgn in max_grad_norm:
    for dr in dropout:
      for hs in hidden_size:
	experiment_name='bidaf_lr_'+str(lr)+'_maxnorm_'+str(mgn)+'_dp_'+str(dr)+'_hidden_'+str(hs)
	print(experiment_name)
	command = 'python2 main.py --experiment_name ' + experiment_name + \
          ' --mode train --num_epochs 10 --learning_rate ' + str(lr) + \
          ' --max_gradient_norm ' + str(mgn) + ' --dropout ' + str(dr) + \
          ' --hidden_size ' + str(hs)
        print (command)
        os.system(command)
