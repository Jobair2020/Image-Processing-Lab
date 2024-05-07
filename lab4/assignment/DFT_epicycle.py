import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import tau
from scipy.integrate import quad_vec
import matplotlib.animation as animation


def imaginary(t, t_list, x_list, y_list):
    return np.interp(t, t_list, x_list + 1j * y_list)


def integrate_function(t):
    return imaginary(t, t_list, x_list, y_list) * np.exp(-n * t * 1j)


def create_path(points):
    len_points = points.shape[0]
    path = np.zeros((len_points, 2), dtype=np.int64)
    path[0] = points[0]
    added = 1
    not_added = points[1:].astype(np.int64)
    vector_lengths = np.zeros((len_points,), dtype=np.float64)

    while added != len_points:
        dist = np.abs(path[added - 1] - not_added)
        r = 5
        empty_circle = True
        vector_len = np.zeros((dist.shape[0],), dtype=np.float64)
        while empty_circle:
            for i in range(dist.shape[0]):
                if dist[i, 0] <= r and dist[i, 1] <= r:
                    empty_circle = False
                    vector_len[i] = np.sqrt(dist[i, 0] ** 2 + dist[i, 1] ** 2)
                else:
                    vector_len[i] = 10000
            r += 5

        min_dist = np.argsort(vector_len)[0]
        path[added] = not_added[min_dist]
        vector_lengths[added] = vector_len[min_dist]
        not_added = np.delete(not_added, min_dist, axis=0)
        added += 1

    return path


img = cv2.imread("face.jpeg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img_gray, 100, 200)
# cv2.imshow("output",edges)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

edge_points = np.argwhere(edges == 255)

connected_path = create_path(edge_points)

x_list, y_list = connected_path[:, 1], -connected_path[:, 0]

plt.plot(x_list, y_list)
plt.show()

x_list = x_list - np.mean(x_list)
y_list = y_list - np.mean(y_list)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_list, y_list)

xlim_data = plt.xlim()
ylim_data = plt.ylim()

plt.show()

t_list = np.linspace(0, tau, len(x_list))  # now we can relate f(t) -> x,y
order = 100

print("calculating the discrete fourier transform for the coefficients ... ")
c = []

for n in range(-order, order + 1):
    # calculate definite integration from 0 to 2*PI
    coef = 1 / tau * quad_vec(integrate_function, 0, tau, limit=100, full_output=1)[0]
    c.append(coef)

c = np.array(c)

draw_x, draw_y = [], []

# make figure for animation
fig, ax = plt.subplots()

circles = [ax.plot([], [], 'r-')[0] for i in range(-order, order + 1)]
# radius
circle_lines = [ax.plot([], [], 'b-')[0] for i in range(-order, order + 1)]
# drawing
drawing, = ax.plot([], [], 'k-', linewidth=2)

# original drawing
orig_drawing, = ax.plot([], [], 'g-', linewidth=0.5)

# to fix the size of figure so that the figure does not get cropped/trimmed
ax.set_xlim(xlim_data[0] - 200, xlim_data[1] + 200)
ax.set_ylim(ylim_data[0] - 200, ylim_data[1] + 200)

ax.set_axis_off()
ax.set_aspect('equal')

print(" Animation creation started ... ")
frames = 300


# save the coefficients in order 0, 1, -1, 2, -2, ...  it is necessary to make epicycles
def sort_coeff(coeffs):
    new_coeffs = []
    new_coeffs.append(coeffs[order])
    for i in range(1, order + 1):
        new_coeffs.extend([coeffs[order + i], coeffs[order - i]])
    return np.array(new_coeffs)


# make frame at time t
# t goes from 0 to 2*PI for complete cycle
def make_frame(i, time, coeffs):
    t = time[i]
    # this is responsible for making rotation of circle
    exp_term = np.array([np.exp(n * t * 1j) for n in range(-order, order + 1)])
    coeffs = sort_coeff(coeffs * exp_term)  # coeffs * exp_term makes the circle rotate.
    # coeffs itself gives only direction and size of circle
    x_coeffs = np.real(coeffs)
    y_coeffs = np.imag(coeffs)
    center_x, center_y = 0, 0

    # make epicycle
    for i, (x_coeff, y_coeff) in enumerate(zip(x_coeffs, y_coeffs)):
        r = np.linalg.norm([x_coeff, y_coeff])  # similar to magnitude: sqrt(x^2+y^2)
        theta = np.linspace(0, tau, num=50)  # theta should go from 0 to 2*PI to get all points of circle
        x, y = center_x + r * np.cos(theta), center_y + r * np.sin(theta)
        circles[i].set_data(x, y)

        # draw a line to indicate the direction of circle
        x, y = [center_x, center_x + x_coeff], [center_y, center_y + y_coeff]
        circle_lines[i].set_data(x, y)

        # calculate center for next circle
        center_x, center_y = center_x + x_coeff, center_y + y_coeff

    draw_x.append(center_x)
    draw_y.append(center_y)

    drawing.set_data(draw_x, draw_y)
    # orig_drawing.set_data(x_list, y_list)


time = np.linspace(0, tau, num=frames)
anim = animation.FuncAnimation(fig, make_frame, frames=frames, fargs=(time, c), interval=5)
plt.show()
# anim.save('epicycle.mp4', writer='ffmpeg', fps=30)
# anim.save('epicycle.gif', writer='pillow', fps=100)
