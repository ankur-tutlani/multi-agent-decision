
import numpy as np   
import pandas as pd
from collections import Counter
import datetime
import random
import matplotlib.pyplot as plt
import networkx as nx
import string
from functools import reduce
import itertools




dotprod = lambda K, L: reduce(lambda z1, z2: z1+z2, map(lambda x: reduce(lambda x1, x2: x1*x2, x), zip(K, L))) if len(K)==len(L) else 0

# 1.random_seed : random seed to reproduce results. Integer
# 2.iteration_name: give any name for this iterations. String
# 3.path_to_save_output. the location where output excel files should be saved. String
# 4.num_strategies : total number of strategies. Integer
# 5.agent_initial_state : specify the initial state of agents in list format e.g. ['H','M','M'] in case of 3 agents where 1st agent follows 'H' strategy and agents 2 and 3 follow 'M'. List
# 6.strategy_share. percent distribution of strategies among agents. Must be specified in list. E.g.[0.4,0.4,0.2]. Must add up to 1. Specify in the order of strategies 0,1,2, and so on. List
# 7.strategy_pair. Strategy pair in dictionary format.E.g. {0: "H",1: "M",2: "L"}. 0 strategy is 'H', 1 strategy is 'M', and so on. Count must match with num_strategies argument. Keys would need to be 0,1,2, etc. Value corresponding to keys can be anything in string format.
# 8.payoff_values. List of payoff values. E.g. [[0,0,70],[0,50,50],[30,30,30]]. the first list [0,0,70] is the first row of the 3*3 payoff matrix assuming we have 3 strategies. Second list [0,50,50] is the second row of the payoff matrix and third list [30,30,30] is the third row of payoff matrix.
## It looks something like below.
#	     [0,  0, 70
#         0, 50, 50
#        30, 30, 30]
# 9.network_name: specify one of these values [small_world1,small_world2,small_world3,complete,random,grid2d]. String
# 10.prob_edge_rewire: small world network parameter. Probability of rewiring each edge. Float
# 11.grid_network_m:  2-dimensional grid network parameter. Number of nodes. Integer
# 12.grid_network_n:  2-dimensional grid network parameter. Number of nodes. Integer
# 13.num_agents : number of agents. Integer
# 14.num_neighbors: number of neighbours. Integer
# 15.delta_to_add: delta payoff value which will be added (or deducted in case of negative value) to the payoffs in case of norm displacement.Float/Integer
# 16.norms_agents_frequency: norm condition. Minimum percentage of agents require to propose same name. Specify number from 0 to 1. Float
# 17.norms_time_frequency: norm condition. Minimum percentage of times agents require to propose same name. Specify number from 0 to 1. Float     
# 18.min_time_period: minimum time period for which the game should be run before adding delta payoff. Integer
# 19.enable_delta_payoff: if norm displacement is required to be assessed , specify 'Yes' else specify 'No'.
# 20.num_of_trials: number of trials. Integer
# 21.fixed_agents: agents assumed as fixed. List
# 22.fixed_strategy_to_use: specify key value from strategy_pair. e.g. if we want fixed agents to follow strategy 'H' and strategy_pair is {0: "H",1: "M",2: "L"}, specify the value as 0. Integer
# 23.function_to_use: specify one of these values [perturbed_response1,perturbed_response2,perturbed_response3,perturbed_response4]. String  
# 24.perturb_ratio: probability of agents taking action randomly. Float



# function_to_use
# perturbed_response1: Agent selects the best response (1-perturb_ratio)*100% times among the strategies which are most frequently used. Agents selects random strategy (perturb_ratio)*100% times from which are not most frequently used. 
# perturbed_response2: Agent selects strategy according to the % share in which it has been used by opponents in the past.
# perturbed_response3: This is same as perturbed_response1 function except agent selects random strategy (perturb_ratio)*100% times from all the strategies. 
# perturbed_response4: Agent selects the best response 100% times among the strategies which are most frequently used. There is no perturbation element.
    
 
# Note there may be instances wherein more than 1 strategy has been used by opponent agents more frequently.
# E.g. if an agent comes across s1 and s2 strategy used by their opponent agents most frequently during any history and both s1 and s2
# have been used equally in the past, in that case agent deciding to take action will select randomly from s1 and s2.


# network_name
# small_world1: Returns a Watts–Strogatz small-world graph. Here number of edges remained constant once we increase the prob_edge_rewire value.Shortcut edges if added would replace the existing ones. But total count of edges remained constant.
# small_world2: Returns a Newman–Watts–Strogatz small-world graph. Here number of edges increased once we increase the prob_edge_rewire value. Would add more shortcut edges in addition to what already exist.
# small_world3: Returns a connected Watts–Strogatz small-world graph.
# complete: Returns the complete graph.
# random: Compute a random graph by swapping edges of a given graph.
# grid2d: Return the 2d grid graph of mxn nodes, each connected to its nearest neighbors.



def perturbed_response1(num_agents,agent_and_strategy_pd2,potential_edges,perturb_ratio,unique_strategies,fixed_agents,fixed_strategy_to_use):
    
    agent_strategy_to_choose=[]

    if len(fixed_agents) > 0:

        for j in range(num_agents):

            if j in fixed_agents:
                agent_strategy_to_choose.append(fixed_strategy_to_use)


            else:

                check1 = [i for i in potential_edges if i[0] == j]
                check2 = [i for i in potential_edges if i[1] == j]
                check1.extend(check2)
                flat_list1 = list(set([item for sublist in check1 for item in sublist]))
                data_to_check = agent_and_strategy_pd2.loc[agent_and_strategy_pd2['agent_no'].isin(flat_list1)].reset_index(drop=True)

                if data_to_check.loc[data_to_check['agent_no']==j]['payoff'].values[0] == max(data_to_check['payoff']):
                    xx = data_to_check.loc[data_to_check['agent_no']==j]['strategy'].values[0]
                else:
                    xx = list(set(data_to_check.loc[(data_to_check['agent_no']!=j) & (data_to_check['payoff']==max(data_to_check['payoff']))]['strategy'].tolist()))
                    xx = random.choices(population=xx,k=1)[0]

                non_recommended = [k for k in unique_strategies if k != xx]

                if len(non_recommended) > 0:
                    draw2= random.choices(population=non_recommended,k=1)
                    best_response = random.choices(population=[xx,draw2[0]],weights=[1-perturb_ratio,perturb_ratio],k=1)
                    best_response = best_response[0]
                else:
                    best_response = xx

                agent_strategy_to_choose.append(best_response)



    else:
        for j in range(num_agents):
            try:
                check1 = [i for i in potential_edges if i[0] == j]
                check2 = [i for i in potential_edges if i[1] == j]
                check1.extend(check2)
                flat_list1 = list(set([item for sublist in check1 for item in sublist]))
                data_to_check = agent_and_strategy_pd2.loc[agent_and_strategy_pd2['agent_no'].isin(flat_list1)].reset_index(drop=True)

                if data_to_check.loc[data_to_check['agent_no']==j]['payoff'].values[0] == max(data_to_check['payoff']):
                    xx = data_to_check.loc[data_to_check['agent_no']==j]['strategy'].values[0]
                else:
                    xx = list(set(data_to_check.loc[(data_to_check['agent_no']!=j) & (data_to_check['payoff']==max(data_to_check['payoff']))]['strategy'].tolist()))
                    xx = random.choices(population=xx,k=1)[0]

                non_recommended = [k for k in unique_strategies if k != xx]

                if len(non_recommended) > 0:
                    draw2= random.choices(population=non_recommended,k=1)
                    best_response = random.choices(population=[xx,draw2[0]],weights=[1-perturb_ratio,perturb_ratio],k=1)
                    best_response = best_response[0]
                else:
                    best_response = xx

                agent_strategy_to_choose.append(best_response)
            
            except:
                
                agent_strategy_to_choose.append(agent_and_strategy_pd2.loc[agent_and_strategy_pd2['agent_no']==j]['strategy'].values[0])


    return agent_strategy_to_choose




def perturbed_response2(num_agents,agent_and_strategy_pd2,potential_edges,perturb_ratio,unique_strategies,fixed_agents,fixed_strategy_to_use):
    
    agent_strategy_to_choose=[]

    if len(fixed_agents) > 0:

        for j in range(num_agents):

            if j in fixed_agents:
                agent_strategy_to_choose.append(fixed_strategy_to_use)
            else:
                check1 = [i for i in potential_edges if i[0] == j]
                check2 = [i for i in potential_edges if i[1] == j]
                check1.extend(check2)
                flat_list1 = list(set([item for sublist in check1 for item in sublist]))
                flat_list1.remove(j) ## to get opponent's agents 


                percent_share_data =agent_and_strategy_pd2.loc[agent_and_strategy_pd2['agent_no'].isin(flat_list1)].reset_index(drop=True)['strategy'].value_counts(normalize=True).to_frame()
                percent_share_data.columns = ['prob']
                percent_share_data['strategy'] = percent_share_data.index.values
                percent_share_data['prob'] = np.round(percent_share_data['prob'],2)
                prob_opponent_strategy = []
                for i in unique_strategies: ### should be in 0,1,2,,,, order 
                    try:
                        xx = percent_share_data.loc[percent_share_data['strategy']==i]['prob'].values[0]
                    except:
                        xx = 0
                    prob_opponent_strategy.append(xx)

                agent_payoffs=[]
                for i in unique_strategies:
                    agent_payoffs.append(dotprod(payoff_matrix[i,].tolist(),prob_opponent_strategy))

                index_strategy = [index for index,item in enumerate(agent_payoffs) if item == max(agent_payoffs)]
                agent_choice = random.choices(index_strategy,k=1)[0]
                agent_strategy_to_choose.append(agent_choice)


    else:
        for j in range(num_agents):
            try:
                
                check1 = [i for i in potential_edges if i[0] == j]
                check2 = [i for i in potential_edges if i[1] == j]
                check1.extend(check2)
                flat_list1 = list(set([item for sublist in check1 for item in sublist]))
                flat_list1.remove(j) ## to get opponent's agents 

                ### this is aggregated data..
                percent_share_data =agent_and_strategy_pd2.loc[agent_and_strategy_pd2['agent_no'].isin(flat_list1)].reset_index(drop=True)['strategy'].value_counts(normalize=True).to_frame()
                percent_share_data.columns = ['prob']
                percent_share_data['strategy'] = percent_share_data.index.values
                percent_share_data['prob'] = np.round(percent_share_data['prob'],2)
                prob_opponent_strategy = []
                for i in unique_strategies: ### should be in 0,1,2,,,, order 
                    try:
                        xx = percent_share_data.loc[percent_share_data['strategy']==i]['prob'].values[0]
                    except:
                        xx = 0
                    prob_opponent_strategy.append(xx)

                agent_payoffs=[]
                for i in unique_strategies:
                    agent_payoffs.append(dotprod(payoff_matrix[i,].tolist(),prob_opponent_strategy))

                index_strategy = [index for index,item in enumerate(agent_payoffs) if item == max(agent_payoffs)]
                agent_choice = random.choices(index_strategy,k=1)[0]
                agent_strategy_to_choose.append(agent_choice)
            
            except:
                
                agent_strategy_to_choose.append(agent_and_strategy_pd2.loc[agent_and_strategy_pd2['agent_no']==j]['strategy'].values[0])

    return agent_strategy_to_choose




def perturbed_response3(num_agents,agent_and_strategy_pd2,potential_edges,perturb_ratio,unique_strategies,fixed_agents,fixed_strategy_to_use):
    
    agent_strategy_to_choose=[]

    if len(fixed_agents) > 0:

        for j in range(num_agents):

            if j in fixed_agents:
                agent_strategy_to_choose.append(fixed_strategy_to_use)
            else:
                check1 = [i for i in potential_edges if i[0] == j]
                check2 = [i for i in potential_edges if i[1] == j]
                check1.extend(check2)
                flat_list1 = list(set([item for sublist in check1 for item in sublist]))
                data_to_check = agent_and_strategy_pd2.loc[agent_and_strategy_pd2['agent_no'].isin(flat_list1)].reset_index(drop=True)

                if data_to_check.loc[data_to_check['agent_no']==j]['payoff'].values[0] == max(data_to_check['payoff']):
                    xx = data_to_check.loc[data_to_check['agent_no']==j]['strategy'].values[0]
                else:
                    xx = list(set(data_to_check.loc[(data_to_check['agent_no']!=j) & (data_to_check['payoff']==max(data_to_check['payoff']))]['strategy'].tolist()))
                    xx = random.choices(population=xx,k=1)[0]


                draw2= random.choices(population=unique_strategies,k=1)
                best_response = random.choices(population=[xx,draw2[0]],weights=[1-perturb_ratio,perturb_ratio],k=1)
                best_response = best_response[0]

                agent_strategy_to_choose.append(best_response)



    else:
        for j in range(num_agents):
            try:
                
                check1 = [i for i in potential_edges if i[0] == j]
                check2 = [i for i in potential_edges if i[1] == j]
                check1.extend(check2)
                flat_list1 = list(set([item for sublist in check1 for item in sublist]))
                data_to_check = agent_and_strategy_pd2.loc[agent_and_strategy_pd2['agent_no'].isin(flat_list1)].reset_index(drop=True)

                if data_to_check.loc[data_to_check['agent_no']==j]['payoff'].values[0] == max(data_to_check['payoff']):
                    xx = data_to_check.loc[data_to_check['agent_no']==j]['strategy'].values[0]
                else:
                    xx = list(set(data_to_check.loc[(data_to_check['agent_no']!=j) & (data_to_check['payoff']==max(data_to_check['payoff']))]['strategy'].tolist()))
                    xx = random.choices(population=xx,k=1)[0]


                draw2= random.choices(population=unique_strategies,k=1)
                best_response = random.choices(population=[xx,draw2[0]],weights=[1-perturb_ratio,perturb_ratio],k=1)
                best_response = best_response[0]

                agent_strategy_to_choose.append(best_response)
            
            except:
                agent_strategy_to_choose.append(agent_and_strategy_pd2.loc[agent_and_strategy_pd2['agent_no']==j]['strategy'].values[0])


    return agent_strategy_to_choose




def perturbed_response4(num_agents,agent_and_strategy_pd2,potential_edges,fixed_agents,fixed_strategy_to_use):
    agent_strategy_to_choose=[]

    if len(fixed_agents) > 0:

        for j in range(num_agents):
            if j in fixed_agents:
                agent_strategy_to_choose.append(fixed_strategy_to_use)

            else:

                check1 = [i for i in potential_edges if i[0] == j]
                check2 = [i for i in potential_edges if i[1] == j]
                check1.extend(check2)
                flat_list1 = list(set([item for sublist in check1 for item in sublist]))
                data_to_check = agent_and_strategy_pd2.loc[agent_and_strategy_pd2['agent_no'].isin(flat_list1)].reset_index(drop=True)

                if data_to_check.loc[data_to_check['agent_no']==j]['payoff'].values[0] == max(data_to_check['payoff']):
                    agent_strategy_to_choose.append(data_to_check.loc[data_to_check['agent_no']==j]['strategy'].values[0])
                else:
                    xx = list(set(data_to_check.loc[(data_to_check['agent_no']!=j) & (data_to_check['payoff']==max(data_to_check['payoff']))]['strategy'].tolist()))
                    agent_strategy_to_choose.append(random.choices(population=xx,k=1)[0])

    else:
        for j in range(num_agents):
            try:
                check1 = [i for i in potential_edges if i[0] == j]
                check2 = [i for i in potential_edges if i[1] == j]
                check1.extend(check2)
                flat_list1 = list(set([item for sublist in check1 for item in sublist]))
                data_to_check = agent_and_strategy_pd2.loc[agent_and_strategy_pd2['agent_no'].isin(flat_list1)].reset_index(drop=True)

                if data_to_check.loc[data_to_check['agent_no']==j]['payoff'].values[0] == max(data_to_check['payoff']):
                    agent_strategy_to_choose.append(data_to_check.loc[data_to_check['agent_no']==j]['strategy'].values[0])
                else:
                    xx = list(set(data_to_check.loc[(data_to_check['agent_no']!=j) & (data_to_check['payoff']==max(data_to_check['payoff']))]['strategy'].tolist()))
                    agent_strategy_to_choose.append(random.choices(population=xx,k=1)[0])
                    
            except:
                agent_strategy_to_choose.append(agent_and_strategy_pd2.loc[agent_and_strategy_pd2['agent_no']==j]['strategy'].values[0])
                
    return agent_strategy_to_choose




def simulation_function_neighbors(random_seed,
                                 iteration_name,
                                 path_to_save_output,
                                 num_strategies,
                                 agent_initial_state,
                                 strategy_share,
                                 strategy_pair,
                                 payoff_values,
                                 network_name,
                                 prob_edge_rewire,
                                 grid_network_m,
                                 grid_network_n,
                                 num_agents, 
                                 num_neighbors,
                                 delta_to_add,
                                 norms_agents_frequency,
                                 norms_time_frequency,
                                 min_time_period,
                                 enable_delta_payoff,
                                 num_of_trials,
                                 fixed_agents,
                                 fixed_strategy_to_use,
                                 function_to_use,
                                 perturb_ratio
                                  
                                 ):
    random_seed = random_seed
    iteration_name = iteration_name
    today = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path_to_save_output = path_to_save_output
    num_strategies = num_strategies
    agent_initial_state = agent_initial_state
    strategy_share = strategy_share
    thisdict = strategy_pair
    random.seed(random_seed)
    payoff_values= payoff_values
    network_name = network_name
    prob_edge_rewire = prob_edge_rewire
    grid_network_m=grid_network_m
    grid_network_n=grid_network_n
    num_agents = num_agents 
    num_neighbors = num_neighbors
    delta_to_add = delta_to_add
    norms_agents_frequency = norms_agents_frequency
    norms_time_frequency = norms_time_frequency
    min_time_period = min_time_period
    enable_delta_payoff = enable_delta_payoff
    num_of_trials =  num_of_trials
    fixed_agents = fixed_agents
    fixed_strategy_to_use = fixed_strategy_to_use
    function_to_use =  function_to_use
    perturb_ratio = perturb_ratio
    
    
    
    payoff_matrix = np.zeros((num_strategies,num_strategies))
    thisdict2 = {value:key for (key,value) in thisdict.items()}
    for i in range(len(payoff_values)):
        payoff_matrix[i,]= payoff_values[i]
        
    list_to_fill = []
    for i in range(num_strategies):
        list_to_fill.append([i]*round(strategy_share[i]*num_agents))
        
    flat_list = [item for sublist in list_to_fill for item in sublist]
    if len(flat_list) == num_agents:
        pass
    
    elif len(flat_list) < num_agents:
        delta = num_agents - len(flat_list)
        for kk in range(delta):
            flat_list.append(flat_list[-1])
   
    else:
        delta = len(flat_list) - num_agents
        for kk in range(delta):
            flat_list.pop()
    

    if len(agent_initial_state) > 0:
        samplelist1 = agent_initial_state
    else:
        samplelist1 = random.sample(flat_list,len(flat_list))
    
    
    
    if network_name == 'small_world1': 
        G = nx.watts_strogatz_graph(num_agents,num_neighbors,prob_edge_rewire)
    if network_name == 'small_world2': 
        G = nx.newman_watts_strogatz_graph(n=num_agents,k=num_neighbors,p=prob_edge_rewire,seed=random_seed)
    if network_name == 'small_world3':
        G = nx.connected_watts_strogatz_graph(n=num_agents,k=num_neighbors,p=prob_edge_rewire,seed=random_seed)
    if network_name == 'complete':
        G = nx.complete_graph(num_agents)
    if network_name == 'random':
        G = nx.watts_strogatz_graph(num_agents,num_neighbors,prob_edge_rewire)
        G = nx.random_reference(G, niter=5, connectivity=True, seed=random_seed)
    if network_name == 'grid2d':
        G = nx.grid_2d_graph(m=grid_network_m,n=grid_network_n)
        mapping = dict(zip(G, range(len(G))))
        G = nx.relabel_nodes(G, mapping)
    
    
    nx.draw(G,with_labels=True)
    plt.savefig(path_to_save_output+"input_network_"+iteration_name+"_"+today+".png")
    plt.clf()
    
    potential_edges = list(G.edges)
    
    agents_in_edges = list(set([i[0] for i in potential_edges]+[i[1] for i in potential_edges]))
    agents_in_network = list(range(num_agents))
    delta_agents_hardcoded = [i for i in agents_in_network if i not in agents_in_edges]
    
    agents_list = list(range(num_agents))
    agent_and_strategy_pd = pd.DataFrame({'agent_no':agents_list,'strategy':samplelist1})
    intial_states_pd = pd.DataFrame([{'initial_state':str([thisdict[k] for k in samplelist1])}],columns=['initial_state'])
    
    
    trend_db_to_store = pd.DataFrame(columns=["percent_count","strategy","timeperiod_number","starting_position"])
    payoff_matrices = []
    payoff_matrices.append(str(payoff_matrix))
    payoff_matrices_timeperiod = []
    payoff_matrices_timeperiod.append(0)
    agent_and_strategy_pd['timeperiod_number'] = 0
    unique_strategies = list(range(num_strategies))
    
    
    
    names_to_check=unique_strategies
    get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
    list_of_colors = get_colors(len(names_to_check))
    color_map = []
    for j in range(len(samplelist1)):
        for i in range(len(names_to_check)):
            if samplelist1[j] == names_to_check[i]:
                color_map.append(list_of_colors[i])

    for i in range(len(G)):
        G.nodes[i]["name_offered"] = [thisdict[k] for k in samplelist1][i]

    labels_for_graph = nx.get_node_attributes(G,"name_offered")
    nx.draw(G,with_labels=True,node_color=color_map,labels=labels_for_graph)
    l,r = plt.xlim()
    plt.xlim(l-0.05,r+0.05)
    plt.savefig(path_to_save_output+"networkgraph_initial_state"+'_'+iteration_name+"_"+today+".png")
    plt.clf()
    
    for timeperiod in range(1,num_of_trials+1):
        payoff_matrix_original = payoff_matrix.copy()

        agent_payoffs_to_store = []  
        for j in range(num_agents):
            try:
                check1 = [i for i in potential_edges if i[0] == j]
                check2 = [i for i in potential_edges if i[1] == j]
                check1.extend(check2)
                flat_list1 = list(set([item for sublist in check1 for item in sublist]))
                flat_list1.remove(j)
                row_strategy = agent_and_strategy_pd.loc[agent_and_strategy_pd['agent_no'] == j]['strategy'].values[0]
                data_to_check = agent_and_strategy_pd.loc[agent_and_strategy_pd['agent_no'].isin(flat_list1)].reset_index(drop=True)
                agent_payoff = 0
                for i in range(len(data_to_check)):
                    agent_payoff += payoff_matrix[row_strategy,data_to_check.iloc[i]['strategy']] 
                agent_payoffs_to_store.append(agent_payoff)
            except:
                agent_payoffs_to_store.append(None)


        agent_and_strategy_payoff = pd.DataFrame({'agent_no':agents_list,'payoff':agent_payoffs_to_store})
        agent_and_strategy_pd2 = pd.merge(agent_and_strategy_pd,agent_and_strategy_payoff,on='agent_no')

        if function_to_use == 'perturbed_response1':
            agent_strategy_to_choose = perturbed_response1(num_agents,agent_and_strategy_pd2,potential_edges,perturb_ratio,unique_strategies,fixed_agents,fixed_strategy_to_use)
        elif function_to_use == 'perturbed_response2':
            agent_strategy_to_choose = perturbed_response2(num_agents,agent_and_strategy_pd2,potential_edges,perturb_ratio,unique_strategies,fixed_agents,fixed_strategy_to_use)
        elif function_to_use == 'perturbed_response3':
            agent_strategy_to_choose = perturbed_response3(num_agents,agent_and_strategy_pd2,potential_edges,perturb_ratio,unique_strategies,fixed_agents,fixed_strategy_to_use)
        elif function_to_use == 'perturbed_response4':
            agent_strategy_to_choose = perturbed_response4(num_agents,agent_and_strategy_pd2,potential_edges,fixed_agents,fixed_strategy_to_use)


        agent_and_strategy_pd= pd.DataFrame({'agent_no':agents_list,'strategy':agent_strategy_to_choose})

        data1_to_store = agent_and_strategy_pd['strategy'].value_counts(normalize=True).to_frame()
        data1_to_store.columns = ['percent_count']
        data1_to_store['strategy'] = data1_to_store.index.tolist()
        data1_to_store = data1_to_store.replace({"strategy":thisdict})
        data1_to_store["timeperiod_number"] = timeperiod
        data1_to_store["starting_position"] = str([thisdict[k] for k in samplelist1])
        trend_db_to_store = pd.concat([trend_db_to_store,data1_to_store],ignore_index=True)

        
        if enable_delta_payoff == 'Yes' and timeperiod > min_time_period and max(trend_db_to_store['percent_count']) >= norms_agents_frequency:
            norms_to_store = []
            percent_time_frequency = []
            norms_candidates = trend_db_to_store.loc[trend_db_to_store['percent_count']>= norms_agents_frequency]
            norms_candidates = norms_candidates.replace({'strategy':thisdict2}) ### convert to numeric.
            potential_norms_candidate = np.unique(norms_candidates['strategy']).tolist()
            for k in potential_norms_candidate:
                norms_candidates2 = norms_candidates.loc[norms_candidates['strategy']==k]
                distinct_timeperiod = len(np.unique(norms_candidates2['timeperiod_number']))
                norms_candidates2 = round(distinct_timeperiod/len(np.unique(trend_db_to_store['timeperiod_number'])),2)
                if norms_candidates2 >= norms_time_frequency:
                    norms_to_store.append(k)
                    percent_time_frequency.append(norms_candidates2)


            if len(norms_to_store) > 0:
                payoff_values_new = []
                norm_strategies = norms_to_store
                for i in range(len(payoff_values)):
                    if i in  norm_strategies:
                        payoff_values_new.append(payoff_values[i])
                    else:
                        payoff_values_new.append([x + delta_to_add for x in payoff_values[i]])

                for i in range(len(payoff_values_new)):  ### replace with new payoff values ###
                    payoff_matrix[i,]= payoff_values_new[i]


                if (payoff_matrix_original==payoff_matrix).all() == False:
                    payoff_matrices.append(str(payoff_matrix))
                    payoff_matrices_timeperiod.append(timeperiod)


    payoff_matrices_pd = pd.DataFrame(payoff_matrices)
    payoff_matrices_pd.columns = ['payoff_matrix']
    payoff_matrices_pd['timeperiod'] = payoff_matrices_timeperiod
    
    
    norms_to_store = []
    percent_time_frequency = []
    norms_candidates = trend_db_to_store.loc[trend_db_to_store['percent_count']>= norms_agents_frequency]
    
    potential_norms_candidate = np.unique(norms_candidates['strategy']).tolist()
    for k in potential_norms_candidate:
        norms_candidates2 = norms_candidates.loc[norms_candidates['strategy']==k]
        distinct_timeperiod = len(np.unique(norms_candidates2['timeperiod_number']))
        norms_candidates2 = round(distinct_timeperiod/len(np.unique(trend_db_to_store['timeperiod_number'])),2)
        if norms_candidates2 >= norms_time_frequency:
            norms_to_store.append(k)
            percent_time_frequency.append(norms_candidates2)
    
    
    try:
        norms_candidates2 = pd.DataFrame()
        norms_candidates2["percent_count"] = percent_time_frequency
        norms_candidates2["name"] = norms_to_store
        if len(norms_candidates2) > 0:
            norms_candidates2.to_excel(path_to_save_output+"normcandidates_"+iteration_name+"_"+today+".xlsx",index=None)
    except:
        pass
    
    
    
    
    color_map = []
    for j in range(len(agent_strategy_to_choose)):
        for i in range(len(names_to_check)):
            if agent_strategy_to_choose[j] == names_to_check[i]:
                color_map.append(list_of_colors[i])

    for i in range(len(G)):
        G.nodes[i]["name_offered"] = [thisdict[k] for k in agent_strategy_to_choose][i]

    labels_for_graph = nx.get_node_attributes(G,"name_offered")
    nx.draw(G,with_labels=True,node_color=color_map,labels=labels_for_graph)
    l,r = plt.xlim()
    plt.xlim(l-0.05,r+0.05)
    
    plt.savefig(path_to_save_output+"network_after_"+str(num_of_trials)+'_timeperiods_'+iteration_name+"_"+today+".png")
    plt.clf()
    
    
    
    fig,ax = plt.subplots()
    data_for_trend_plot = trend_db_to_store.loc[trend_db_to_store['starting_position']==str([thisdict[k] for k in samplelist1])]
    data_for_trend_plot = data_for_trend_plot.reset_index(drop=True)
    for label,grp in data_for_trend_plot.groupby('strategy'):
        grp.plot(x='timeperiod_number',y='percent_count',ax=ax,label=label)
    
    
    x_values11 = data_for_trend_plot['timeperiod_number'].unique()
    total_periods11 = len(x_values11)
    max_display_labels = 5  
    interval11 = max(1, total_periods11 // max_display_labels)  

    ax.set_xticks(np.arange(len(x_values11))[::interval11])
    ax.set_xticklabels(x_values11[::interval11])
    
    
    ax.set_xlabel('Timeperiod')
    ax.set_ylabel('Count %')
    
    plt.savefig(path_to_save_output+"strategy_trend_"+iteration_name+"_"+today+".png")
    plt.clf()
    
    
    
    db_to_fill2 = pd.DataFrame()
    db_to_fill2["timeperiod"] = -1
    db_to_fill2["name_offered"] = -1
    
    if len(norms_to_store) > 0:
        for j in norms_to_store:
            foocheck = data_for_trend_plot.loc[data_for_trend_plot["strategy"]==j]
            foocheck = foocheck.loc[foocheck['percent_count']>= norms_agents_frequency].reset_index(drop=True)
            foocheck = foocheck.sort_values(["timeperiod_number"])
            foocheck["count_names_offered"] = (foocheck["strategy"]==j).cumsum()
            foocheck["cum_perc"] = foocheck["count_names_offered"]/len(np.unique(data_for_trend_plot['timeperiod_number']))
            xxxx= foocheck.loc[foocheck["cum_perc"]>=norms_time_frequency][["timeperiod_number"]].head(1)
            if xxxx.shape[0] > 0:
                timev = foocheck.loc[foocheck["cum_perc"]>=norms_time_frequency][["timeperiod_number"]].head(1)["timeperiod_number"].values[0]
                foodb = pd.DataFrame({"timeperiod":[timev],"name_offered":[j]})
                db_to_fill2 = pd.concat([db_to_fill2,foodb],ignore_index=True,sort=False)
                
    
    
    try:
        if len(db_to_fill2) > 0:
            db_to_fill2.to_excel(path_to_save_output+"time_when_reached_norm_"+iteration_name+"_"+today+".xlsx",index=None)
    except:
        pass
    
    
    trend_db_to_store.to_excel(path_to_save_output+"aggregate_data_detailed_agent_"+iteration_name+"_"+today+".xlsx",index=None)

    
    parameters_pd = pd.DataFrame([{'random_seed':random_seed,'iteration_name':iteration_name,
                              'path_to_save_output':path_to_save_output,'num_strategies':num_strategies,
                              'agent_initial_state':str(agent_initial_state),'strategy_share':str(strategy_share),
                              'strategy_pair':str(strategy_pair),'payoff_values':str(payoff_values),
                              'network_name':network_name,'prob_edge_rewire':prob_edge_rewire,
                              'grid_network_m':grid_network_m,'grid_network_n':grid_network_n,
                              'num_agents':num_agents,'num_neighbors':num_neighbors,
                              'delta_to_add':delta_to_add,'norms_agents_frequency':norms_agents_frequency,
                              'norms_time_frequency':norms_time_frequency,'min_time_period':min_time_period,
                              'enable_delta_payoff':enable_delta_payoff,'num_of_trials':num_of_trials,
                              'fixed_agents':str(fixed_agents),'fixed_strategy_to_use':fixed_strategy_to_use,
                              'function_to_use':function_to_use,'perturb_ratio':perturb_ratio,
                              'datetime':today}]).T
    parameters_pd.columns=["parameter_values"]
    parameters_pd["parameter"]=parameters_pd.index
    parameters_pd[["parameter","parameter_values"]].to_excel(path_to_save_output+"parameters_"+iteration_name+"_"+today+".xlsx",index=None)
    
    intial_states_pd.to_excel(path_to_save_output+"initial_state_considered_"+iteration_name+"_"+today+".xlsx",index=None)
    payoff_matrices_pd.to_excel(path_to_save_output+"payoff_matrices_considered_by_timeperiod_"+iteration_name+"_"+today+".xlsx",index=None)

    return(print("done"))

