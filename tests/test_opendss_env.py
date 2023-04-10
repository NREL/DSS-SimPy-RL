# testing open dss env
import os
import sys
directory = os.path.dirname(os.path.realpath(__file__))
desktop_path = os.path.dirname(os.path.dirname(directory))
sys.path.insert(0,desktop_path+'\ARM_IRL')
from envs.openDSSenv import openDSSenv
import opendssdirect as dss
import random


der = True

dss_data_dir = desktop_path+'\\ARM_IRL\\cases\\123Bus_Simple\\'
dss_master_file_dir = 'Redirect ' + dss_data_dir + 'IEEE123Master.dss'

if der:
    dss_data_dir = desktop_path+'\\ARM_IRL\\cases\\123Bus_SimpleMod\\'
    dss_master_file_dir = 'Redirect ' + dss_data_dir + 'IEEE123Master.dss'

dss.run_command(dss_master_file_dir)
circuit = dss.Circuit
#critical_loads_bus = ['58','59','99','100','88','93','94','78','48','50', '111','114', '37','39']
critical_loads_bus = ['58','59','99','100','88','94','78','48','50', '111','114', '37','39']
capacitor_banks =['C83', 'C88a', 'C90b','C92c']
# switch from and two buses, with the first 6 are normally closed and the last two are normally open
switches = { 0: ['150r','149'], 1: ['13','152'], 2: ['18','135'], 3: ['60','160'], 4: ['97','197'], 5: ['61','61s'], 6: ['151','300'], 7: ['54','94'] }

switch_names =[]
for k,sw in enumerate(switches):
    switch_names.append('Sw'+str(k+1))

#line_faults = ['L55','L68', 'L58', 'L77', 'L45', 'L101', 'L41']
line_faults = ['L55', 'L58', 'L77','L68','L45', 'L101','L41']

env = openDSSenv(_dss = dss, _critical_loads=critical_loads_bus, _line_faults =line_faults, _switch_names = switch_names, _capacitor_banks = capacitor_banks,debug=True,contingency=2)

episodes = 100
max_episode_len = 50

for i in range(episodes):
    print('Episode {}'.format(i+1))
    state = env.reset()
    print('observation : {}'.format(state))

    done = False
    ctr = 0
    episodic_reward = 0
    switch_selected = []
    while not done and ctr < max_episode_len:
        
        # randomly select an action for time-being until we train an agent
        action = random.choice(switch_names[0:])
        ctr+=1
        if action in switch_selected:
            continue
        else:
            switch_selected.append(action)
        
        
        #print('Switch closing '+action)
        next_state, reward, done,info,_ = env.step(action, result={})
        state = next_state
        episodic_reward += reward[0]
    print('Episode Length : '+str(ctr))
    




