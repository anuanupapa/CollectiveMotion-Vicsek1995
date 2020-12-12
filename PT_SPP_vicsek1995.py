import numpy as np
import numba as nb
import matplotlib as mpl
import matplotlib.animation
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use("Agg")



#Makes the particles move in the direction of v
@nb.njit
def move(pos_ini, v, dt):
    pos_final = pos_ini + v*dt
    return(pos_final)



#Finds the neighbours after calculating the euclidean distances
@nb.njit
def find_neighbours(fp_pos, all_pos, radius, N):
    distances = np.zeros((N,1))
    for pInd in range(len(all_pos)):
        distances[pInd] = np.linalg.norm(fp_pos - all_pos[pInd])
    #return neighbours within radius
    neighbours = np.where(distances <= radius)
    return(neighbours[0])



#Determines the direction in which the particle will be heading
@nb.njit
def find_orientation(neighbours, thetas, noise):
    theta_new = np.sum(thetas[neighbours])/len(neighbours)
    #mean orientation in the neighbourhood + noise
    return(theta_new + noise*np.random.random())



#find the motion for only one particle
#Putting the neighbour finding and theta calculation together
#uses find_neighbours() and find_orientation()
@nb.njit
def update_angle(pos_focal, pos_all, theta_all, radius, N,
           noise=0.1):
    neigh = find_neighbours(pos_focal, pos_all, radius, N)
    thetanew = find_orientation(neigh, theta_all, noise)
    return(thetanew)


#updates velocity vectors
#uses update_angle()
@nb.njit
def update_velocity(pos_focal, pos_all, theta_all, radius, N):
    theta = update_angle(pos_focal, pos_all,
                         theta_all, radius, N)
    vy = v*np.sin(theta)
    vx = v*np.cos(theta)
    return(vx, vy, theta)



#Update the direction of all the particles
#uses update_velocity()
@nb.njit
def run_1step(N, L, theta_all, pos_all, vel_all, radius, dt):

    theta_old = np.copy(theta_all)
    pos_old = np.copy(pos_all)
    vel_old = np.copy(vel_all)

    vel_new = np.zeros((N,2))
    theta_new = np.zeros((N,1))
    pos_new = np.zeros((N,2))
    
    #for loop reqiured to find the new orientations
    for n in range(N):
        [vx, vy, thet] = update_velocity(pos_old[n], pos_old,
                                         theta_old, radius, N)
        vel_new[n][0] = vx
        vel_new[n][1] = vy
        theta_new[n] = thet
        [x, y] = move(pos_old[n], vel_new[n], dt)
        pos_new[n][0] = x%(L)
        pos_new[n][1] = y%(L)
        
    return(theta_new, pos_new, vel_new)

        

def initialize(N, L, v):
    positions = np.zeros((N,2))
    velocities = np.zeros((N,2))
    thetas = np.zeros((N,1))
    for i in range(N):
        positions[i][0] = (L/5) - (L/20) + np.random.random()*(L/10)
        positions[i][1] = (L/5) - (L/20) + np.random.random()*(L/10)
        velx = -v + np.random.random()*2*v
        velocities[i][0] = velx
        vely = (-1)**(np.random.random()<0.5)*np.sqrt(v*v - velx*velx)
        velocities[i][1] = vely
        velocities[i] = np.random.permutation(velocities[i])
        thetas[i] = np.arctan(velx/vely)
    return(positions, velocities, thetas)


#make movie of the particles
def movie_update(frame, pos_arr, mov_arr):
    plt.clf()
    plt.quiver(position_evolution_arr[frame, :, 0], 
               position_evolution_arr[frame, :, 1],
               velocity_evolution_arr[frame, :, 0], 
               velocity_evolution_arr[frame, :, 1])
    
    
    

if __name__ == "__main__":

    n = 300
    l = 250
    v = 1.0
    radius = 1
    dt = 1
    t = 10000
    
    [positions, velocities, thetas] = initialize(n, l, v)

    position_evolution_arr = np.zeros((t+1, n, 2))
    position_evolution_arr[0] = positions

    velocity_evolution_arr = np.zeros((t+1, n, 2))
    velocity_evolution_arr[0] = velocities
    
    [pos_arr, vel_arr, theta_arr] = initialize(n, l, v)
    
    for ti in range(t):
        print(ti)
        [thetas, positions, velocities] = run_1step(n, l, thetas,
                                                    positions,
                                                    velocities,
                                                    radius, dt)
        #print(np.shape(positions))
        #print(np.shape(velocities))
        position_evolution_arr[ti + 1] = positions
        velocity_evolution_arr[ti + 1] = velocities

    print(np.shape(velocity_evolution_arr))
    print(np.shape(position_evolution_arr))
    
    plt.quiver(position_evolution_arr[0, :, 0],
               position_evolution_arr[0, :, 1],
               velocity_evolution_arr[0, :, 0], 
               velocity_evolution_arr[0, :, 1])
    plt.show()
    
    plt.quiver(position_evolution_arr[t, :, 0], 
               position_evolution_arr[t, :, 1],
               velocity_evolution_arr[t, :, 0], 
               velocity_evolution_arr[t, :, 1])
    plt.show()


    fig, ax = plt.subplots()
    Writer = matplotlib.animation.writers['ffmpeg']
    writer = Writer(fps=10,
                    metadata=dict(artist='Anuran'), bitrate=1800)

    ani =matplotlib.animation.FuncAnimation(fig, movie_update,
                                            fargs = [position_evolution_arr, velocity_evolution_arr],
                                            frames=np.arange(0,100,1),
                                            interval=10, repeat=False)
    ani.save('CollectiveMotion_1flock.mp4', writer=writer)
    #plt.show()


    
