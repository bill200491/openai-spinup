import time
import joblib
import os
import gym
import os.path as osp
import tensorflow as tf
import numpy as np
import core
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.gail_dataset import continue_Dset

def load_policy(fpath, itr='last', deterministic=False):

    # handle which epoch to load from
    if itr=='last':
        saves = [int(x[11:]) for x in os.listdir(fpath) if 'simple_save' in x and len(x)>11]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d'%itr

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, osp.join(fpath, 'simple_save'+itr))

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action


def behavior_clone( env_id, traj_fn, ac_kwargs, actor_critic, lr=1e-3, itr=1000, deterministic=False):

    #seed = 10000 * proc_id()
    #tf.set_random_seed(seed)
    #np.random.seed(seed)

    env = gym.make( env_id)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    
    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph = core.placeholders_from_spaces(env.observation_space, env.action_space)
    l_ph = core.placeholder()
    adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)

    # Main outputs from computation graph
    pi, mu, logp, logp_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)

    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi]
    
    # Action loss
    loss_op = tf.reduce_mean(tf.square( a_ph - mu))
    #loss_op = tf.reduce_mean(tf.square( l_ph - logp_pi))
    
    
    
    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = mu
    
    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={ x_ph: x[None,:]})[0]
    
    # Optimizers
    train_op = MpiAdamOptimizer(learning_rate=lr).minimize(loss_op)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Sync params across processes
    sess.run(sync_all_params())

    # load trajectories
    dataset = continue_Dset(expert_path= traj_fn, traj_limitation=-1)
    
    batch_size = 128
    train_batchs= 1+dataset.train_set.num_pairs// batch_size
    valid_batchs= 1+dataset.val_set.num_pairs// batch_size
    print('--->',itr, batch_size, train_batchs, valid_batchs)

    for i in range(itr):
        train_loss=0
        for j in range(train_batchs):
            ob_expert, ac_expert = dataset.get_next_batch( batch_size, 'train')
            #lg_expert= sess.run(logp, {x_ph: ob_expert, a_ph:ac_expert} )
            #loss= sess.run([train_op, loss_op], {x_ph: ob_expert, l_ph:lg_expert})[-1:]
            loss= sess.run([train_op, loss_op], {x_ph: ob_expert, a_ph:ac_expert})[-1:]
            train_loss += loss[0]
        train_loss /= train_batchs
        
        valid_loss=0
        for j in range(valid_batchs):
            ob_expert, ac_expert = dataset.get_next_batch( batch_size, 'val')
            #lg_expert= sess.run(logp, {x_ph: ob_expert, a_ph:ac_expert} )
            #loss= sess.run([train_op, loss_op], {x_ph: ob_expert, l_ph:lg_expert})[-1:]
            loss= sess.run([train_op, loss_op], {x_ph: ob_expert, a_ph:ac_expert})[-1:]
            valid_loss += loss[0]
        valid_loss /= valid_batchs
        
        print('%d Loss - train: %0.3f  valid:%9.3f'%(i, train_loss, valid_loss))
        
    
    #exit()
    return env, get_action

    

def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


# pkill -9 ^python3 ; python3 bc.py BipedalWalker-v2 traj-100.npz --hid 500 --l 3 --itr=10


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_id', type=str)
    parser.add_argument('traj', type=str)
    parser.add_argument('--len', type=int, default=0)
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action = behavior_clone( args.env_id, args.traj,
                                     dict(hidden_sizes=[args.hid]*args.l),
                                    core.mlp_actor_critic, itr= args.itr)
    run_policy(env, get_action, args.len, args.episodes, not(args.norender))
    
    
