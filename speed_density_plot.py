import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', size=18)
plt.rc('ps', useafm=True)
plt.rc('pdf', use14corefonts=True)

# car length (meters)
car_len = 5

def get_max_possible_speed(car_spacing, time_spacing):
    return (car_spacing * car_len) * 18 / (time_spacing * 5)

def get_density(speed, time_spacing):
    # speed is in kmph
    car_spacing = (speed * time_spacing * 5) / (car_len * 18)
    return (1 / (1+car_spacing))

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.set_title('Estimated speed-density curve')
ax.set_xlabel('Link density', labelpad=5)
ax.set_ylabel('Speed (kmph)', labelpad=5)

# c = np.asarray([5, 4, 3, 2, 1, 0.5])

# density = 1 / (1 + car_spacing)
# x = 1/(1+c) 

# car_spacing = (1 / density) - 1
# c = 1/x - 1

# starting point (speed = 100 kmph, assuming that's the speed limit)
x0_t06 = get_density(100, 0.6)
x0_t09 = get_density(100, 0.9)
x0_t20 = get_density(100, 2)
x0_t30 = get_density(100, 3)

d1 = np.linspace(x0_t06, 0.9, 1000)
d2 = np.linspace(x0_t09, 0.9, 1000)
d3 = np.linspace(x0_t20, 0.9, 1000)
d4 = np.linspace(x0_t30, 0.9, 1000)

y1 = get_max_possible_speed((1/d1) - 1, 0.6)
y2 = get_max_possible_speed((1/d2) - 1, 0.9)
y3 = get_max_possible_speed((1/d3) - 1, 2)
y4 = get_max_possible_speed((1/d4) - 1, 3)

obj1, = ax.plot(d1, y1, label='0.6-s spacing', lw=2)
obj2, = ax.plot(d2, y2, label='0.9-s spacing', lw=2)
obj3, = ax.plot(d3, y3, label='2-s spacing', lw=2)
obj4, = ax.plot(d4, y4, label='3-s spacing', lw=2)

# show the s1, s2 and slim
#xi = 0.5
xi = 0.545
yi = get_max_possible_speed((1/xi) - 1, 0.9)
print(get_max_possible_speed((1/xi) - 1, 0.6))
print(get_max_possible_speed((1/xi) - 1, 0.9))
ax.plot(xi, yi, 'k.', ms=10)
ax.vlines(xi, 0, yi, colors='k', linestyles='dashed')
ax.hlines(yi, 0, xi, colors='k', linestyles='dashed')
ax.text(xi+0.02, yi+0.02, r'$s_1$ = ({:.2f}, {:.0f} kmph)'.format(xi, yi), fontsize='x-small', fontweight='bold')

ax.hlines(100, 0, get_density(100, 0.6), colors='k', linestyles='dashed')
ax.text(get_density(100, 0.6) + 0.02, 100, 'NYC highway speed limit', verticalalignment='center', fontsize='x-small', fontweight='bold')

ax.hlines(50, 0, get_density(50, 0.6), colors='k', linestyles='dashed')
ax.text(get_density(50, 0.6) + 0.02, 50, 'Nairobi city speed limit', verticalalignment='center', fontsize='x-small', fontweight='bold')

ax.set_xlim(0, 1)
ax.set_ylim(ymin=0)

#ax.fill_between(x, y3, y1, color='k', alpha=0.1)

ax.grid()
fig.legend((obj1, obj2, obj3, obj4), (obj1.get_label(), obj2.get_label(), obj3.get_label(), obj4.get_label()), ncol=4, loc='upper center', fontsize='x-small', borderpad=0.4, handlelength=1.0, handletextpad=0.6, columnspacing=1.0)

ax.tick_params(direction='out', length=8)
fig.subplots_adjust(left=0.18, right=0.96, bottom=0.18, top=0.88)

fig.savefig('../figures/speed-density-plot.png')
fig.savefig('../figures/speed-density-plot.pdf')

plt.show()

plt.close(fig)
