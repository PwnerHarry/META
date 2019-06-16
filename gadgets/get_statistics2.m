function X = get_statistics2(results, num_points, log_flag)
if log_flag || isempty(log_flag)
episodes = size(results, 2);
ratio = nthroot(episodes, num_points - 1);
X = max(ones(1, num_points), round(ratio .^ (0: num_points - 1)));
else
    episodes = size(results, 2);
    X = max(ones(size(linspace(0, 1, num_points))), round(episodes * linspace(0, 1, num_points)));
end
end