function sample = simplify_data(name, results, num_points)
sample.name = name;
[sample.X, sample.MEAN, sample.INTERVAL] = get_statistics(results, num_points);
band_drawer(sample.X, sample.MEAN, sample.INTERVAL);
set(gca, 'xscale', 'log');
set(gca, 'yscale', 'log');
filename = sprintf('frozenlake/%s.mat', name);
save(filename, 'sample');
end