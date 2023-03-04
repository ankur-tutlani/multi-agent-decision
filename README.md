
# Decision-making under Multi-Agent Systems and Social Networks

This library is used to ascertain how agents take decisions under multi-agent systems. Agents are connected with other agents via a social network and play symmetric games. Agents take decisions based upon their interactions with other agents in the neighbourhood. The strategy which is being played most frequently by agents and for a longer duration of time is a potential candidate for being called a norm. This library demonstrates how norms evolve from bottom-up when agents interact with other agents and decide what is best for them.


## How to use it?


```bash
  # Install
  pip install multi-agent-decision

  
  # Import
  from multi_agent_decision import simulation

  # Execute 
  simulation.simulation_function_neighbors(random_seed = 226822,
                                 iteration_name = "test1",
                                 path_to_save_output = "C:\\Users\\Downloads\\",
                                 num_strategies=3,
                                 agent_initial_state =[],
                                 strategy_share = [0.4,0.4,0.2],
                                 strategy_pair = {0: "H",1: "M",2: "L"},
                                 payoff_values = [[0,0,70],[0,50,50],[30,30,30]],
                                 network_name = "small_world1",
                                 prob_edge_rewire = 0.50,
                                 grid_network_m = 5,
                                 grid_network_n = 8,
                                 num_agents = 20, 
                                 num_neighbors = 2,
                                 delta_to_add = 20,
                                 norms_agents_frequency = 0.7,
                                 norms_time_frequency = 0.5,
                                 min_time_period = 20,
                                 enable_delta_payoff = "Yes",
                                 num_of_trials = 50,
                                 fixed_agents = [5,9],
                                 fixed_strategy_to_use = 0,
                                 function_to_use = "perturbed_response4",
                                 perturb_ratio = 0.05
                                  
                                 )


```
    
## Function parameters
Following are the parameters which are required to be specified. At the end in the parenthesis, it shows the data type of the parameter which is required or the possible values which is required to be used. In following we will use agents' strategies, actions, choices, responses interchangeably. These all represent same intent and meaning in our context.  

1. random_seed : Random seed to reproduce results. (Integer)
2. iteration_name: Give any name for this iteration. (String)
3. path_to_save_output : The location where output excel files should be saved. (String)
4. num_strategies : Total number of strategies. (Integer)
5. agent_initial_state : Specify the initial state of agents in list format e.g. ["H","M","M"] in case of 3 agents where 1st agent follows "H" strategy and agents 2 and 3 follow "M". (List)
6. strategy_share : Percent distribution of strategies among agents. Must be specified in list format. E.g.[0.4,0.4,0.2]. Must add up to 1. Specify in the order of strategies 0,1,2, and so on. (List)
7. strategy_pair : Strategy pair in dictionary format. E.g. {0: "H",1: "M",2: "L"}. 0 strategy is "H", 1 strategy is "M", and so on. Count must match with num_strategies argument. Keys would need to be 0,1,2, etc. Value corresponding to keys can be anything in string format. (Dict)
8. payoff_values : List of payoff values. E.g. [[0,0,70],[0,50,50],[30,30,30]]. The first list [0,0,70] is the first row of the 3*3 payoff matrix assuming we have 3 strategies. Second list [0,50,50] is the second row of the payoff matrix and third list [30,30,30] is the third row of payoff matrix.(List)
9. network_name : Specify one of these values. A detailed explanation is provided below. ["small_world1","small_world2","small_world3","complete","random","grid2d"]. (String)
10. prob_edge_rewire : Small world network parameter. Probability of rewiring existing edges or adding new edges. (Float)
11. grid_network_m :  2-dimensional grid network parameter. Number of nodes. (Integer)
12. grid_network_n :  2-dimensional grid network parameter. Number of nodes. (Integer)
13. num_agents : Number of agents. (Integer)
14. num_neighbors : Number of neighbours. (Integer)
15. delta_to_add : Delta payoff value which will be added (or deducted in case of negative value) to the payoffs in case of norm displacement. (Float/Integer)
16. norms_agents_frequency : Norm condition. Minimum percentage of agents required to use same strategy. Specify number from 0 (0% agents) to 1(100% agents). (Float)
17. norms_time_frequency : Norm condition. Minimum percentage of times agents require to use same strategy. Specify number from 0 (no time period) to 1 (all time periods). (Float)
18. min_time_period : Minimum time period for which the game should be run before adding delta payoff. (Integer)
19. enable_delta_payoff : If norm displacement is required to be assessed , specify "Yes" else specify "No".
20. num_of_trials : Number of trials, how long game should run. (Integer)
21. fixed_agents : Agents assumed as fixed. e.g. [2,3] shows we want agents 2 and 3 to be fixed. (List)
22. fixed_strategy_to_use : Specify key value from strategy_pair. e.g. if we want fixed agents to follow strategy "H" and strategy_pair is {0: "H",1: "M",2: "L"}, specify the value as 0. (Integer)
23. function_to_use : Specify one of these values. A detailed explanation is provided below. ["perturbed_response1","perturbed_response2","perturbed_response3","perturbed_response4"]. (String)
24. perturb_ratio : Probability of agents taking action randomly. (Float)


<br />

In the "function_to_use" parameter above, below is the explanation for what these different values mean inside the list specified above.

1. perturbed_response1: Agents select the best response (1-perturb_ratio)*100% times among the strategies which are most frequently used. If there is more than one such strategy, then agents select any one strategy randomly. Agents select random strategy (perturb_ratio)*100% times among the strategies which are not most frequently used. Here also, if there is more than one such strategy, then agents select any one strategy randomly out of these.
2. perturbed_response2: Agent selects strategy randomly according to the relative weightage of different strategies. These relative weights are the % of times the respective strategy has been used by opponents in the past. 
3. perturbed_response3: This is same as "perturbed_response1" function except agent selects random strategy (perturb_ratio)*100% times from all the possible strategies and not only from the ones which were not used most frequently by their opponents.
4. perturbed_response4: Agent selects the best response 100% of times among the strategies which are most frequently used. If there is more than one such strategy, then agents select any one strategy randomly. There is no perturbation element (perturb_ratio is considered as zero).


<br/>

In the "network_name" parameter above, below is the explanation for what these different values mean inside the list specified above.
1. small_world1: Returns a Watts Strogatz small-world graph. Here number of edges remained constant once we increase the prob_edge_rewire value. Shortcut edges if added would replace the existing ones. But total count of edges remained constant.
2. small_world2: Returns a Newman Watts Strogatz small-world graph. Here number of edges increased once we increase the prob_edge_rewire value. It would add more shortcut edges in addition to what already exist.
3. small_world3: Returns a connected Watts Strogatz small-world graph. Rest of the explanation remains as small_world1.
4. complete: Returns the complete graph.
5. random: Compute a random graph by swapping edges of a given graph. The given graph used is Watts Strogatz small-world graph (the one produced by "small_world1").
6. grid2d: Return the 2d grid graph of mxn nodes, each connected to its nearest neighbors.

We have used the networkx python library to populate these graphs. For more information around these graphs and how these are produced please refer to the link in the reference section.



## Function explanation
Here we explain the underlying functioning of this library with the help of an example. But this can be replicated to any symmetric game wherein we have defined finite strategies along with its payoff values.

We assume there is a Nash demand game wherein each agent can make 3 demands, High (70), Medium (50) or Low (30). The rule of the game is if the total demands made by two agents in any given iteration are more than 100, no one is going to get anything. Below is the payoff matrix of row versus column player looks like.

		  H	  M	    L
    H	(0,0)	(0,0)	 (70,30)
    M	(0,0)	(50,50)	 (50,30)
    L	(30,70)	(30,50)	 (30,30)
	

There are 3 pure strategy Nash equilibria, (H, L), (M, M) and (L, H). There is not a strictly or weakly dominant strategy for any of the row or column player. Suppose that agents are connected in a ring network wherein each agent is connected with two other agents, one on the left and one on the right like in below image.


![](https://github.com/ankur-tutlani/multi-agent-decision/raw/main/network_graph.png)


Agents update their strategy if any of their neighbours (defined by "num_neighbors") are having higher payoff else, they stick to their own strategy which they have been following. Payoffs are computed as follows. If agent with strategy H meets agent with strategy M and L, then payoff from meeting with strategy M agent equals 0 and payoff from meeting with strategy L agent equals 70, hence the total payoff equals 70. In case of a tie, meaning payoffs are exactly same following current strategy versus payoffs from neighbours" strategy then agents stick to the strategy which they have been following. 

We assume agents are distributed in a specific ratio in a network. E.g., if total agents ("num_agents") are 20 and if they are distributed in 40/40/20 distribution ("strategy_share"), that implies there are 8 H type agents, 8 M type agents and 4 L type agents. This can be represented in a list format like this [H,H,H,H,H,H,H,H,M,M,M,M,M,M,M,M,L,L,L,L]. This list shows first agent follows "H" (first value from the left), last agent follows "L" (last value from the right), agent 16 follows "M" etc.This list can be rearranged in multiple ways but still satisfies the condition of 40/40/20. Hence, we require to specify initial state of the agents in "agent_initial_state" parameter in the list format. If this parameter is empty, the function will select any one state randomly. This initial state is being shown in "initial_state_considered_..."  excel file output. For more details on the output generated, please refer to the section below.

We start the game with one edge of the network is selected at any given point. Edge of the network is represented as (0,1), (5,4) etc, implying agents 0 and 1 are connected with each other, agents 5 and 4 are connected with each other. During each time period, all edges are selected sequentially and the agents associated with those edges play the game. We are interested in knowing how the initial state of H/M/L distribution changes as agents interact over a period of time and update their strategy. And how these results change when agents start the game with different initial states (e.g. more H type agents placed with H types or more H type agents placed with M type etc.).

We have considered different combinations of strategies that agents can adopt while taking actions. We assume that agents use perturbed best response implying agents can take action randomly out of H,M or L with a certain positive probability (Young, 2015). At each time when agents require to take action, they look at the strategies of their neighbours and calculate their payoff. All agents calculate their payoffs in the current period and decide if they want to change strategies or stick to what they have been following so far. We have tested four different ways which agents can use to decide what action to take. The "function_to_use" parameter provides details about these and how these are different to each other.

When the simulations are run for "num_of_trials" timeperiod, we get the percentage distribution of  H/M/L which agents used. This shows the percentage of agents who used H/M or L during each time period. The strategy which satisfy the two conditions for norm specified by "norms_agents_frequency" and "norms_time_frequency" are considered as norm. We have looked at two dimensions of norms, number of agents following the norm and for how long it has been followed. We can see the output like below. In below graph, X-axis shows the timeperiod, and Y-axis shows the % of agents who played the respective strategy. Y-axis values are in ratio format (range from 0 - 1), so would need to multiply by 100 to get this in percentage format. 


![](https://github.com/ankur-tutlani/multi-agent-decision/raw/main/norm_emergence.png)



In above figure, strategy "M" satisfies the norm criteria, when we assume "norms_agents_frequency" as 0.7 and "norms_time_frequency" as 0.5. This implies at least 70% of agents following "M" for at least 50% of times. The detailed explanation of the output generated is provided in the next section.

We have also considered the possibility of norm displacement. This is controlled by three parameters, "enable_delta_payoff", "delta_to_add", and "min_time_period" parameters. For norm displacement, we need to specify "enable_delta_payoff" value to "Yes" and also specify a positive/negative value for "delta_to_add" parameter. The "min_time_period" parameter specifies the minimum time period for which the game should run before checking for norm displacement. Its value should be set lower than "num_of_trials" parameter. When any strategy satisfies the norm criteria laid out (say "M"), then the "delta_to_add" value is being added to the payoff values for rest of the other strategies ("H" and "L") after "min_time_period". The revised payoff matrix would be used by agents once the delta value is added to the payoff values. This would continue as long as some other strategy satisfies the norm criteria and the game is still not played for "num_of_trials" timeperiods. If some other strategy now satisfies the norm criteria (say "H"), then the delta payoff value would be added to the rest of the strategies ("M" and "L"). And this process continues till the game is played for all the timeperiods ("num_of_trials"). This can be seen in below figure where "M" strategy is being displaced by "H" strategy after around 40 timeperiods. And then again there is an attempt to displace "H" strategy with "M" strategy later in the game at around 90th timeperiod.


![](https://github.com/ankur-tutlani/multi-agent-decision/raw/main/norm_displacement.png)



## How to interpret output?
There are total 10 files generated in the output when there is at least 1 strategy which satisfies the norm criteria.

    input_network_ringnetwork_2023-02-02-21-31-34.png
    networkgraph_initial_state_ringnetwork_2023-02-02-21-31-34.png
    strategy_trend_ringnetwork_2023-02-02-21-31-34.png
	network_after_100_timeperiods_ringnetwork_2023-02-02-21-31-34.png
    normcandidates_ringnetwork_2023-02-02-21-31-34.xlsx
    time_when_reached_norm_ringnetwork_2023-02-02-21-31-34.xlsx
	parameters_ringnetwork_2023-02-02-21-31-34.xlsx
	aggregate_data_detailed_agent_ringnetwork_2023-02-02-21-31-34.xlsx
    initial_state_considered_ringnetwork_2023-02-02-21-31-34.xlsx
	payoff_matrices_considered_by_timeperiod_ringnetwork_2023-02-02-21-31-34.xlsx
    


The 4 image files (.png) contain the network graphs and trend graph which is being considered for the simulation. The "input_network_.." file is the input network with which the game is started. "networkgraph_initial_state_...." file shows initial strategy distribution in the ratio specified in "strategy_share" parameter. The "strategy_trend_..." file shows the percent of agents following "H","M" or "L" strategy at each time period during the game. The "network_after_..." file is the network state at the end of game. Agents following the same strategy would be coloured in the same colour in this file.

Norm file "normcandidates_..." shows strategy that satisfies the norm criteria laid out. File "time_when_reached_norm_..." shows the time period number when the specific strategy met the norm criteria. These 2 files are generated only when at least one strategy satisfies the norm criteria. 

Parameters file "parameters_.." lists all the parameters which have been specified in the function call. File "aggregate_data_detailed_.." has the information on percentage of agents who proposed different strategies at each time period during the game. "initial_state_considered_" file shows the initial state with which game is started. In case of norm displacement, "payoff_matrices_considered_.." shows the payoff matrix considered along with time period. When there is more than one row in this file, this shows payoff matrix is being revised during the simulation run. The timeperiod column in this file shows the time period point at which the payoff matrix is revised.

All the file names end with date and time stamp when the function was executed. It also contains information on network structure used like "ringnetwork" or "smallworld" network etc.


## References
1. Young, H.P. "The evolution of social norms" Annual Review of Economics. 2015 (7),pp.359-387
2. https://pypi.org/project/networkx/ 

