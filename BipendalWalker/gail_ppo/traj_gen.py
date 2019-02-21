import time
import joblib
import os
import os.path as osp
import tensorflow as tf
import numpy as np
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph

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


def generate_trajectory(env, logger, get_action, max_ep_len=None,  render=True):
    # Initialize history arrays
    obs = []
    rews = []
    news = []
    acs = []
    
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    d= True # Followed by baselines 
    
    while True:
        #if render:
            #env.render()
            #time.sleep(1e-3)

        a = get_action(o)
        
        obs.append( o.tolist())
        news.append( d)
        acs.append( a.tolist())
        
        o, r, d, _ = env.step(a)
        rews.append( r)
        
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            break

    obs = np.array(obs)
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)
    print( obs.shape, acs.shape)
          
    if 280.0< ep_ret:
        traj = {"ob": obs, "rew": rews, "new": news, "ac": acs,
                "ep_ret": ep_ret, "ep_len": ep_len}
    else:
        traj = {"ep_ret": ep_ret, "ep_len": ep_len}
    return traj
    


def run_policy(env, traj_fn, get_action, max_ep_len=None, num_episodes=100, render=True):
    
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    obs_list = []
    acs_list = []
    len_list = []
    ret_list = []
    logger = EpochLogger()
    
    n=0
    while n < num_episodes:
        traj = generate_trajectory(env, logger, get_action, max_ep_len=max_ep_len,  render=num_episodes)
        print('Episode %d \t EpRet %.3f \t EpLen %d'%( n, traj['ep_ret'], traj['ep_len']))
        
        if 'ob' in traj:
            obs, acs, ep_len, ep_ret = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret']
            obs_list.append(obs)
            acs_list.append(acs)
            len_list.append(ep_len)
            ret_list.append(ep_ret)
            n +=1
    A= np.array(obs_list)
    B= np.array(acs_list)
    print('----->', A.shape, B.shape, '---', A[0].shape, B[0].shape)
    np.savez( traj_fn, obs=np.array(obs_list), acs=np.array(acs_list), ep_lens=np.array(len_list), ep_rets=np.array(ret_list))
    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()
    
    

# pkill -9 ^python3 ;  python3 traj_gen.py ../../../data/BipedalWalker/BipedalWalker_ppo03/ traj-10 -d -n 10 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('traj_fn', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action = load_policy(args.fpath, 
                                  args.itr if args.itr >=0 else 'last',
                                  args.deterministic)
    run_policy(env, args.traj_fn, get_action, args.len, args.episodes, not(args.norender))
    
    
