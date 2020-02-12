function [CURVE, BAND] = band_drawer(varargin)

set(gcf, 'Renderer', 'Painter');
if nargin == 3 % X, MEAN, INTERVAL
    X = varargin{1};
    MEAN = varargin{2};
    INTERVAL = varargin{3};
    LineWidth = 1;
    Curve = plot(X, MEAN, 'LineWidth', LineWidth);
elseif nargin == 4 % X, MEAN, INTERVAL, COLOR, LineWidth
    X = varargin{1};
    MEAN = varargin{2};
    INTERVAL = varargin{3};
    COLOR = varargin{4};
    LineWidth = 1;
    Curve = plot(X, MEAN, 'LineWidth', LineWidth, 'COLOR', COLOR);
elseif nargin == 5 % X, MEAN, INTERVAL, COLOR, LineWidth
    X = varargin{1};
    MEAN = varargin{2};
    INTERVAL = varargin{3};
    COLOR = varargin{4};
    LineWidth = varargin{5};
    Curve = plot(X, MEAN, 'LineWidth', LineWidth, 'COLOR', COLOR);
end
COLOR = Curve.Color;
delete(Curve);
BAND = fill([X; flip(X)], [INTERVAL(:, 1); flip(INTERVAL(:, 2))], COLOR, 'EdgeColor', 'none', 'FaceAlpha', '0.2');
hold on;
CURVE = plot(X, MEAN, '-', 'LineWidth', LineWidth, 'COLOR', COLOR);
drawnow;
end