from utils import *
from methods import *
from white import direct_greedy

runtimes, episodes, N, gamma = int(50), int(1e3), 11, lambda x: 0.95
env, decide, observation_to_x, policy = init_ring_world(N, p = [0.5, 0.5])
true_value = dynamic_programming(env, policy, gamma=gamma)

# dggtd_results = eval_method(direct_greedy, env, decide, observation_to_x, true_value, N = N, runtimes=runtimes, episodes=episodes)
# ggtd_results = eval_method(greedy, env, decide, observation_to_x, true_value, N = N, runtimes=runtimes, episodes=episodes)
# gtd_results = eval_method(gtd, env, decide, observation_to_x, true_value, N = N, runtimes=runtimes, episodes=episodes)
# togtd_results = eval_method(true_online_gtd, env, decide, observation_to_x, true_value, N = N, runtimes=runtimes, episodes=episodes)
# mc_results = eval_method(monte_carlo, env, decide, observation_to_x, true_value, N = N, runtimes=runtimes, episodes=episodes)

# plot_results(dggtd_results, label='Direct Greedy GTD')
# plot_results(ggtd_results, label='Greedy GTD')
# plot_results(gtd_results, label='GTD(1)')
# plot_results(togtd_results, label='True Online GTD(1)')
# plot_results(mc_results, label='MC')

# plot_results(toggtd_results, label='True Online Greedy GTD')

plt.yscale('log'); plt.show()
pass