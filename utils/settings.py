import math

#step decay
def step_decay(epoch):
   initial_lrate = 0.1
   drop = 0.2
   epochs_drop = 2.0
   lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   return lrate

def time_decay(epoch, initial_lrate):
    decay_rate = 0.01
    new_lrate = initial_lrate/(1+decay_rate*epoch)
    return new_lrate