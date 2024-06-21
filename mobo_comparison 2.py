#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/AI4Finance-Foundation/FinRL-Meta/blob/master/tutorials/1-Introduction/FinRL_PortfolioAllocation_NeurIPS_2020.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Multi-objective Bayesian optimization of Deep Reinforcement Learning algorithms for Stock Trading from Scratch: Portfolio Allocation
# 
# We are going to optimize the hyperparameters of the PPO and SAC algorithms with respect to conflictive risk-performance targets (which for the example can be the Sharpe ratio or Annual return) for simplicity, we can get rid of the ESG objective in the first run and can be just the stability or annual volatility.
# 
# In order to do so, we are going to use FinRL-StableBaselines3 for DRL and for multi-objective Bayesian optimization we can use the package Dragonfly, which contains examples of managing conflictive objectives (https://github.com/dragonfly/dragonfly/tree/master/examples/detailed_use_cases).
# 
# Conflictive objectives for example are performance and risk.
# 
# The training period is going to be: 2008-2022.
# The test period is going to be: 2023.
# 
# We will compare the performance of multi-objective Bayesian optimization with interpolated single objective Bayesian optimization and with Random Search. We will consider 50 iterations to refine the hyperparameters and a low (relative) number of timesteps to make the experiments feasible (unless you want them better, which we can do it with GPUs). The experiments will compare the performance of 10 different runs of every method to account for the noise of the target variables.
# 
# We need to think about how to compare the performance of single vs multi-objective optimization. Multi-objective optimization will return a set of solutions such that they are optimal for both objectives (called the Pareto set), however we need to sample one solution of that set to be compared with the best solution found by single objective methods.
# 
# Further experiments would require the comparison of these methods with respect to genetic algorithms, more indexes (like NASDAQ) more DRL algorithms (like A3C) and more repetitions.  
# 
# 

# # Content

# * [1. Problem Definition](#0)
# * [2. Getting Started - Load Python packages](#1)
#     * [2.1. Install Packages](#1.1)    
#     * [2.2. Check Additional Packages](#1.2)
#     * [2.3. Import Packages](#1.3)
#     * [2.4. Create Folders](#1.4)
# * [3. Download Data](#2)
# * [4. Preprocess Data](#3)        
#     * [4.1. Technical Indicators](#3.1)
#     * [4.2. Perform Feature Engineering](#3.2)
# * [5.Build Environment](#4)  
#     * [5.1. Training & Trade Data Split](#4.1)
#     * [5.2. User-defined Environment](#4.2)   
#     * [5.3. Initialize Environment](#4.3)    
# * [6.Implement DRL Algorithms](#5)  
# * [7.Backtesting Performance](#6)  
#     * [7.1. BackTestStats](#6.1)
#     * [7.2. BackTestPlot](#6.2)   
#     * [7.3. Baseline Stats](#6.3)   
#     * [7.3. Compare to Stock Market Index](#6.4)             

# In[ ]:


# <a id='0'></a>
# # Part 1. Problem Definition

# This problem is to design an automated trading solution for portfolio alloacation. We model the stock trading process as a Markov Decision Process (MDP). We then formulate our trading goal as a maximization problem.
# 
# The algorithm is trained using Deep Reinforcement Learning (DRL) algorithms and the components of the reinforcement learning environment are:
# 
# 
# * Action: The action space describes the allowed actions that the agent interacts with the
# environment. Normally, a ∈ A represents the weight of a stock in the porfolio: a ∈ (-1,1). Assume our stock pool includes N stocks, we can use a list [a<sub>1</sub>, a<sub>2</sub>, ... , a<sub>N</sub>] to determine the weight for each stock in the porfotlio, where a<sub>i</sub> ∈ (-1,1), a<sub>1</sub>+ a<sub>2</sub>+...+a<sub>N</sub>=1. For example, "The weight of AAPL in the portfolio is 10%." is [0.1 , ...].
# 
# * Reward function: r(s, a, s′) is the incentive mechanism for an agent to learn a better action. The change of the portfolio value when action a is taken at state s and arriving at new state s',  i.e., r(s, a, s′) = v′ − v, where v′ and v represent the portfolio
# values at state s′ and s, respectively
# 
# * State: The state space describes the observations that the agent receives from the environment. Just as a human trader needs to analyze various information before executing a trade, so
# our trading agent observes many different features to better learn in an interactive environment.
# 
# * Environment: Dow 30 consituents
# 
# 
# The data of the single stock that we will be using for this case study is obtained from Yahoo Finance API. The data contains Open-High-Low-Close price and volume.
# 

# <a id='1'></a>
# # Part 2. Getting Started- Load Python Packages

# In[ ]:


# <a id='1.1'></a>
# ## 2.1. Install all the packages through FinRL library
# 

# 
# <a id='1.2'></a>
# ## 2.2. Check if the additional packages needed are present, if not install them.
# * Yahoo Finance API
# * pandas
# * numpy
# * matplotlib
# * stockstats
# * OpenAI gym
# * stable-baselines
# * tensorflow
# * pyfolio

# <a id='1.3'></a>
# ## 2.3. Import Packages

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime

from finrl import config
from finrl import config_tickers
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline,convert_daily_return_to_pyfolio_ts
from finrl.meta.data_processor import DataProcessor
from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor
import sys
sys.path.append("../FinRL-Library")


# <a id='1.4'></a>
# ## 2.4. Create Folders

# In[ ]:


import os
if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)


# <a id='2'></a>
# # Part 3. Download Data
# Yahoo Finance is a website that provides stock data, financial news, financial reports, etc. All the data provided by Yahoo Finance is free.
# * FinRL uses a class **YahooDownloader** to fetch data from Yahoo Finance API
# * Call Limit: Using the Public API (without authentication), you are limited to 2,000 requests per hour per IP (or up to a total of 48,000 requests a day).
# 

# In[ ]:


print(config_tickers.DOW_30_TICKER)


# In[ ]:


# Download and save the data in a pandas DataFrame:
#Ojo, esto hay que cambiarlo para que nos adaptemos a un nuevo mercado y fechas.
#Podemos usar factset en vez de Yahoo, que va mucho mejor.
df = YahooDownloader(start_date = '2008-01-01',
                     end_date = '2023-12-31',
                     ticker_list = config_tickers.DOW_30_TICKER).fetch_data()


# In[ ]:


df.head()


# In[ ]:


df.shape


# # Part 4: Preprocess Data
# Data preprocessing is a crucial step for training a high quality machine learning model. We need to check for missing data and do feature engineering in order to convert the data into a model-ready state.
# * Add technical indicators. In practical trading, various information needs to be taken into account, for example the historical stock prices, current holding shares, technical indicators, etc. In this article, we demonstrate two trend-following technical indicators: MACD and RSI.
# * Add turbulence index. Risk-aversion reflects whether an investor will choose to preserve the capital. It also influences one's trading strategy when facing different market volatility level. To control the risk in highly volatile markets, such as financial crisis of 2007–2008, FinRL employs the financial turbulence index that measures extreme asset price fluctuation in order to XXXXX.

# In[ ]:


#Aquí también se pueden introducir mejoras.
fe = FeatureEngineer(
                    use_technical_indicator=True,
                    use_turbulence=False,
                    user_defined_feature = False)

df = fe.preprocess_data(df)


# In[ ]:



# In[ ]:


# add covariance matrix as states
df=df.sort_values(['date','tic'],ignore_index=True)
df.index = df.date.factorize()[0]

cov_list = []
return_list = []

# look back is one year
lookback=252
for i in range(lookback,len(df.index.unique())):
  data_lookback = df.loc[i-lookback:i,:]
  price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
  return_lookback = price_lookback.pct_change().dropna()
  return_list.append(return_lookback)

  covs = return_lookback.cov().values
  cov_list.append(covs)


df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
df = df.merge(df_cov, on='date')
df = df.sort_values(['date','tic']).reset_index(drop=True)

# # Part 5. Design Environment
# Considering the stochastic and interactive nature of the automated stock trading tasks, a financial task is modeled as a **Markov Decision Process (MDP)** problem. The training process involves observing stock price change, taking an action and reward's calculation to have the agent adjusting its strategy accordingly. By interacting with the environment, the trading agent will derive a trading strategy with the maximized rewards as time proceeds.
# 
# Our trading environments, based on OpenAI Gym framework, simulate live stock markets with real market data according to the principle of time-driven simulation.
# 

# In[ ]:


import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv


# ## Training data split: 2008-01-01 to 2022-12-31
# Se debería hacer 10 Fold CV temporal para mejorar a nivel de empresa.

# In[ ]:


train = data_split(df, '2008-01-01','2022-12-31')


# Here is the definition of the environment.

# In[ ]:


class StockPortfolioEnv(gym.Env):
    """A single stock trading environment for OpenAI gym

    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount : int
            start money
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            a threshold to control risk aversion
        day: int
            an increment number to control date

    Methods
    -------
    _sell_stock()
        perform sell action based on the sign of the action
    _buy_stock()
        perform buy action based on the sign of the action
    step()
        at each step the agent will return actions, then
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step


    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                df,
                stock_dim,
                hmax,
                initial_amount,
                transaction_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                turbulence_threshold=None,
                lookback=252,
                day = 0):
        #super(StockEnv, self).__init__()
        #money = 10 , scope = 1
        self.day = day
        self.lookback=lookback
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct =transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low = 0, high = 1,shape = (self.action_space,))
        # Shape = (34, 30)
        # covariance matrix + technical indicators + ESG (4). Ojo, no funciona meter aqui el shape bueno. Esto puede causar problemas.

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space+len(self.tech_indicator_list), self.state_space))

        # load data from a pandas dataframe
        self.data = self.df.loc[self.day,:]
        self.covs = self.data['cov_list'].values[0]

        self.state = np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)


        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        self.date_memory=[self.data.date.unique()[0]]


    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique())-1
        # print(actions)

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ['daily_return']
            plt.plot(df.daily_return.cumsum(),'r')
            plt.savefig('results/cumulative_reward.png')
            plt.close()

            plt.plot(self.portfolio_return_memory,'r')
            plt.savefig('results/rewards.png')
            plt.close()

            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))
            print("end_total_asset:{}".format(self.portfolio_value))

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ['daily_return']
            if df_daily_return['daily_return'].std() !=0:
              sharpe = (252**0.5)*df_daily_return['daily_return'].mean()/ \
                       df_daily_return['daily_return'].std()
              print("Sharpe: ",sharpe)
            print("=================================")

            return self.state, self.reward, self.terminal,{}

        else:
            #print("Model actions: ",actions)
            # actions are the portfolio weight
            # normalize to sum of 1
            #if (np.array(actions) - np.array(actions).min()).sum() != 0:
            #  norm_actions = (np.array(actions) - np.array(actions).min()) / (np.array(actions) - np.array(actions).min()).sum()
            #else:
            #  norm_actions = actions
            weights = self.softmax_normalization(actions)
            #print("Normalized actions: ", weights)
            self.actions_memory.append(weights)
            last_day_memory = self.data

            #load next state
            self.day += 1
            self.data = self.df.loc[self.day,:]
            self.covs = self.data['cov_list'].values[0]
            self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
            #print(self.state)

            portfolio_return = sum(((self.data.close.values / last_day_memory.close.values)-1)*weights)

            #...Weights tbc by investor´s preference
            # portfolio_return = sum(((self.data.close.values / last_day_memory.close.values)-1)*weights)
            # update portfolio value
            new_portfolio_value = self.portfolio_value*(1+portfolio_return)

            #Aqui es donde hay que ponderar el ESG.
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(new_portfolio_value)

            # the reward is the new portfolio value or end portfolo value
            self.reward = new_portfolio_value
            #print("Step reward: ", self.reward)
            #self.reward = self.reward*self.reward_scaling

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        # load states
        self.covs = self.data['cov_list'].values[0]
        self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)

        self.portfolio_value = self.initial_amount
        #self.cost = 0
        #self.trades = 0
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        self.date_memory=[self.data.date.unique()[0]]
        return self.state

    def render(self, mode='human'):
        return self.state

    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator/denominator
        return softmax_output


    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        #print(len(date_list))
        #print(len(asset_list))
        df_account_value = pd.DataFrame({'date':date_list,'daily_return':portfolio_return})
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs


# In[ ]:


stock_dimension = len(train.tic.unique())
state_space = stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")


# In[ ]:


env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "transaction_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": config.INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}

e_train_gym = StockPortfolioEnv(df = train, **env_kwargs)


# In[ ]:


env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))


# <a id='5'></a>
# # Part 6: Implement DRL Algorithms
# BASELINE: If you do standard optimization, you end having only one black-box, which is very problematic to optimize, as the only objective is an interpolation of two conflictive objectives, then we have only a solution which is not very good for both objectives independently, hence we expect the baseline to perform worse that our code.

# ### Model 1: **PPO** multi-objective hyperparameter search
# 

# In[ ]:


# Bayesian optimization baseline and random search blackbox

# In[ ]:


from pyfolio import timeseries

#Vanilla interpolated single objective Bayesian optimization baseline.
def interpolated_financial_portfolio_optimization_baseline(x):#(ent_coef, learning_rate, gamma, clip_range, gae_lambda):
    ent_coef = x[0]
    learning_rate = x[1]
    gamma = x[2]
    clip_range = x[3]
    gae_lambda = x[4]
    timesteps = 10 #Beta value for timesteps. For the experiments, needs to be higher (+-80000)

    agent = DRLAgent(env = env_train)
    PPO_PARAMS = {
      "n_steps": 2048,
      "ent_coef": ent_coef,
      "learning_rate": learning_rate,
      "batch_size": 128,
      "gamma" : gamma,
      "clip_range" : clip_range,
      "gae_lambda" : gae_lambda
    }

    model_ppo = agent.get_model("ppo", model_kwargs = PPO_PARAMS)
    trained_ppo = agent.train_model(model=model_ppo,
                             tb_log_name='ppo',
                             total_timesteps=timesteps)

    trade = data_split(df,'2023-01-01', '2023-12-31')
    e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)
    df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_ppo, environment = e_trade_gym)

    DRL_strat = convert_daily_return_to_pyfolio_ts(df_daily_return)
    perf_func = timeseries.perf_stats
    perf_stats_all = perf_func(returns=DRL_strat,
                              factor_returns=DRL_strat,
                                positions=None, transactions=None, turnover_denom="AGB")
    sharpe_ratio = perf_stats_all["Sharpe ratio"]
    max_drawdown = perf_stats_all["Max drawdown"]
    return sharpe_ratio + 30.0 * max_drawdown


# Bayesian optimization black-boxes: targets

# In[ ]:


def financial_portfolio_optimization_black_box_targets(x):#(ent_coef, learning_rate, gamma, clip_range, gae_lambda):
    timesteps = 10 #Beta value for timesteps. For the experiments, needs to be higher (+-80000)
    ent_coef = x[0]
    learning_rate = x[1]
    gamma = x[2]
    clip_range = x[3]
    gae_lambda = x[4]

    agent = DRLAgent(env = env_train)
    PPO_PARAMS = {
      "n_steps": 2048,
      "ent_coef": ent_coef,
      "learning_rate": learning_rate,
      "batch_size": 128,
      "gamma" : gamma,
      "clip_range" : clip_range,
      "gae_lambda" : gae_lambda
    }

    model_ppo = agent.get_model("ppo", model_kwargs = PPO_PARAMS)
    trained_ppo = agent.train_model(model=model_ppo,
                             tb_log_name='ppo',
                             total_timesteps=timesteps)

    trade = data_split(df,'2023-01-01', '2023-12-31')
    e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)
    df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_ppo, environment = e_trade_gym)

    DRL_strat = convert_daily_return_to_pyfolio_ts(df_daily_return)
    perf_func = timeseries.perf_stats
    perf_stats_all = perf_func(returns=DRL_strat,
                              factor_returns=DRL_strat,
                                positions=None, transactions=None, turnover_denom="AGB")

    return [perf_stats_all["Sharpe ratio"], perf_stats_all["Max drawdown"]]


# Experiment launcher

# In[ ]:

from dragonfly import maximise_function, load_config_file
import random as random
from dragonfly import multiobjective_maximise_functions

MOBO = 'ppo_mobo'
BASELINE = 'ppo_baseline'

_CHOOSER_DICT = {
  'ppo_mobo': (financial_portfolio_optimization_black_box_targets, 'config.json', None),
  'ppo_baseline': (interpolated_financial_portfolio_optimization_baseline, 'config.json', None),
  }

objective, config_file, mf_cost = _CHOOSER_DICT[MOBO]
objective_baseline, config_file, mf_cost = _CHOOSER_DICT[BASELINE]
config = load_config_file(config_file)
number_repetitions = 2 #Must be 25 for real experiments.
seeds = np.linspace(0, number_repetitions-1, number_repetitions).astype(int)
methods = {"MOBO": 0, "BO_BASELINE": 1, "RS" : 2}
n_iters = 2 #Must be 50 in real experiments.
results_experiment = np.zeros([len(methods.keys()), len(seeds), 2]) #Methods, experiments, objs.

ppo_hp_bounds = {'ent_coef': (0.0, 0.1),
                 'learning_rate': (0.000001, 0.1),
                 "gamma" : (0.9, 0.9999),
                 "clip_range": (0.1, 0.3),
                 "gae_lambda" : (0.9, 0.999)}

constraint_limit = -0.09 #Minimum max drawdown. Must be changed to a higher value.

for seed in seeds:

  #Multi-objective Bayesian optimization.
  #Independent black boxes modelled by GPs.
  #To be debugged. It is going to fail.
  pareto_opt_vals, pareto_opt_pts, history = multiobjective_maximise_functions((objective, 2), config.domain, n_iters, config=config)

  #We have the full Pareto set but will just sample one value of the Pareto set.
  #We will only report the last result of the last iteration. 
  results_experiment[methods["MOBO"], seed] = pareto_opt_vals[0] 

  #Interpolate Bayesian optimization baseline.
  opt_val, opt_pt, history = maximise_function(objective_baseline, config.domain,
                                               n_iters, config=config)

  #Here you only have the aggregate between the two objectives.
  results_experiment[methods["BO_BASELINE"], seed] = [opt_val, opt_val] 

  #Random search.
  for iter in range(n_iters):
    ent_coef_random = random.uniform(0.0, 0.1)
    learning_rate_random = random.uniform(0.000001, 0.1)
    gamma_random = random.uniform(0.9, 0.9999)
    clip_range_random = random.uniform(0.1, 0.3)
    gae_lambda = random.uniform(0.9, 0.999)
    result_random = interpolated_financial_portfolio_optimization_baseline([ent_coef_random, learning_rate_random, gamma_random, clip_range_random, gae_lambda])
    results_experiment[methods["RS"], seed] = result_random 

  print("Iteration #" + str(seed) + " done.")

#Comparativa: injusta, ya que el multiobjetivo no esta igual que el resto, pero
#eso lo dejamos para la reunion para ver lo que podemos poner aqui.
means_first_objective = results_experiment[:, :, 0].mean(axis=1)
means_second_objective = results_experiment[:, :, 1].mean(axis=1)
print(means_first_objective)
print(means_second_objective)
