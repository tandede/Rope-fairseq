ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9
tensorboard --logdir=tensorboard_logs --port=6007 --bind_all