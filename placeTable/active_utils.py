# -*- coding: utf-8 -*-
"""
Created on Tue May  8 20:40:43 2018

@author: daniel
"""

import numpy as np
import matplotlib.pyplot as plt
class RbfComplexReward():


    def __init__(self, obj_centers, obj_weights, abs_weights):
        self.offset = 0.1
        self.obj_offsets = np.array([[0.0, 0.0],[-self.offset, self.offset],[self.offset, self.offset],
                                     [-self.offset,-self.offset],[self.offset,-self.offset]])
        #each object should have 5 weights associated with it
        #w0 is center weight, w1 is top left, w2 is top right, w3 is bottom left, w4 is bottom right
        assert(len(obj_centers)*5 == len(obj_weights))


        self.num_objects = obj_centers.shape[0]
        self.num_obj_weights = len(obj_weights)
        self.num_abs_weights = len(abs_weights)
        self.obj_weights = obj_weights
        self.abs_weights = abs_weights
        self.obj_centers = obj_centers

        #need to transform object centers into centers for all rbfs (5 per object)
        self.centers = []
        for center in obj_centers:
            self.centers.extend(np.ones((5,1))*center + self.obj_offsets)

        self.weights = []
        self.weights.extend(obj_weights)
        self.widths = np.tile(np.array([0.3, 0.1, 0.1, 0.1, 0.1]),len(obj_centers))

        #add absolute basis functions too!
        #params for absolute gridding of table 3x3 grid
        self.num_rows = 3
        self.num_cols = 3
        assert(len(abs_weights) == self.num_rows * self.num_cols)
        self.weights.extend(abs_weights)
        self.weights = np.array(self.weights)
        xgrid = np.linspace(0,1,self.num_cols)
        ygrid = np.linspace(1,0, self.num_rows)
        grid_pairs = [[xi,yi] for yi in ygrid for xi in xgrid]
        #print grid_pairs
        self.centers.extend(grid_pairs)
        self.centers = np.array(self.centers)
        table_pos_width = 0.3
        self.widths = np.append(self.widths, np.repeat(table_pos_width, self.num_rows * self.num_cols))



    def get_num_rbfs(self):
        return self.centers.shape[0]

    def gaussian_rbf(self, x, center, width):
        return np.exp(-np.inner((x - center),(x - center)) / width)

    def get_reward(self, x):
        rbf_weighted_sum = 0.0
        for i in range(len(self.centers)):
            rbf_weighted_sum += self.weights[i] * self.gaussian_rbf(x, self.centers[i], self.widths[i])
        return rbf_weighted_sum

    def rbf_heat(self, X,Y):
        heats = np.zeros(X.shape)
        for i in range(len(heats)):
            for j in range(len(heats[0])):
                heats[i,j] = self.get_reward(np.array([X[i,j],Y[i,j]]))
        return heats


    def get_heat_map(self):
        plt.figure()
        #create a grid sampling and plot rbf activations
        granularity = 50
        xspace = np.linspace(0,1,granularity)
        yspace = np.linspace(1,0,granularity)
        pairs = [[(xi,yi) for xi in xspace] for yi in yspace]
        #print pairs
        heatmap = [[self.get_reward(pair) for pair in row] for row in pairs]
        #print heatmap
        #a = np.array([[1,2],[0,0]])
        #plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.imshow(heatmap, cmap='hot')
        plt.colorbar()
        #plt.show()

    def plot_heat_map(self):
        plt.figure()
        n = 50
        x = np.linspace(0., 1., n)
        y = np.linspace(1., 0., n)
        X, Y = np.meshgrid(x, y)

        Z = self.rbf_heat(X,Y)

        plt.pcolormesh(X, Y, Z, cmap = 'hot')
        #plt.imshow(Z, cmap='hot')
        #plt.show()



    def calc_gradient(self,x):
        rbf_grad = 0.0
        for i in range(len(self.centers)):
            deriv_power = (-2.0 * x + 2.0 * self.centers[i]) / self.widths[i]
            rbf_grad += self.weights[i] * self.gaussian_rbf(x, self.centers[i], self.widths[i]) * deriv_power
        return rbf_grad

    def estimate_best_placement(self, num_restarts=10, step_size=0.05, steps=50, plot=False):
        best_reward = -np.inf
        best_placement = np.array([0, 0])
        for rep in range(num_restarts):
            #print rep
            x = np.random.rand(2)
            #plt.plot(x[0],x[1],'o')
            for i in range(steps):
                x += step_size * self.calc_gradient(x)
                #clamp to grid boundaries
                x[x<0.0] = 0.0
                x[x>1.0] = 1.0
                if plot:
                    plt.plot(x[0],x[1],'o')
                #print x
            reward = self.get_reward(x)
            if reward > best_reward:
                best_reward = reward
                best_placement = x
            #plt.plot(x[0],x[1],'o')
            #print x
        #plt.plot(best_placement[0], best_placement[1],'*',markersize=30)
        return best_placement, best_reward

class RbfReward():
    def __init__(self, centers, init_weights, init_widths):
        self.centers = centers
        self.weights = init_weights
        self.widths = init_widths

    def gaussian_rbf(self, x, center, width):
        return np.exp(-np.inner((x - center),(x - center)) / width)

    def get_reward(self, x):
        rbf_weighted_sum = 0.0
        for i in range(len(self.centers)):
            rbf_weighted_sum += self.weights[i] * self.gaussian_rbf(x, self.centers[i], self.widths[i])
        return rbf_weighted_sum

    def rbf_heat(self, X,Y):
        heats = np.zeros(X.shape)
        for i in range(len(heats)):
            for j in range(len(heats[0])):
                heats[i,j] = self.get_reward(np.array([X[i,j],Y[i,j]]))
        return heats


    def get_heat_map(self):
        plt.figure()
        #create a grid sampling and plot rbf activations
        granularity = 50
        xspace = np.linspace(0,1,granularity)
        yspace = np.linspace(1,0,granularity)
        pairs = [[(xi,yi) for xi in xspace] for yi in yspace]
        #print pairs
        heatmap = [[self.get_reward(pair) for pair in row] for row in pairs]
        #print heatmap
        a = np.array([[1,2],[0,0]])
        #plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.imshow(heatmap, cmap='hot')
        plt.colorbar()
        #plt.show()

    def plot_heat_map(self):
        plt.figure()
        n = 50
        x = np.linspace(0., 1., n)
        y = np.linspace(1., 0., n)
        X, Y = np.meshgrid(x, y)

        Z = self.rbf_heat(X,Y)

        plt.pcolormesh(X, Y, Z, cmap = 'hot')
        #plt.imshow(Z, cmap='hot')
        #plt.show()



    def calc_gradient(self,x):
        rbf_grad = 0.0
        for i in range(len(self.centers)):
            deriv_power = (-2.0 * x + 2.0 * self.centers[i]) / self.widths[i]
            rbf_grad += self.weights[i] * self.gaussian_rbf(x, self.centers[i], self.widths[i]) * deriv_power
        return rbf_grad

    def estimate_best_placement(self, num_restarts=50, step_size=0.01, steps=100, plot=False):
        best_reward = -np.inf
        best_placement = np.array([0, 0])
        for rep in range(num_restarts):
            #print rep
            x = np.random.rand(2)
            #plt.plot(x[0],x[1],'o')
            for i in range(steps):
                x += step_size * self.calc_gradient(x)
                #clamp to grid boundaries
                x[x<0.0] = 0.0
                x[x>1.0] = 1.0
                if plot:
                    plt.plot(x[0],x[1],'o')
                #print x
            reward = self.get_reward(x)
            if reward > best_reward:
                best_reward = reward
                best_placement = x
            #plt.plot(x[0],x[1],'o')
            #print x
        #plt.plot(best_placement[0], best_placement[1],'*',markersize=30)
        return best_placement, best_reward

def visualize_reward(rbf, title=""):
    rbf.plot_heat_map()

    best_x, reward = rbf.estimate_best_placement()
    plt.plot(best_x[0], best_x[1],'*',markersize=30)
    print("best x", best_x, "best reward", reward)

    #plot centers of objects
    for c in rbf.obj_centers:
        plt.plot(c[0],c[1],'o',markersize=20)
    plt.title(title)
    #plt.show()

#used to plot between example
def plot_between_reward_positive():


    #assumes grid in [0,1]x[0,1] region

    x = np.array([0., 0.])
    #weights = np.array([1.0, 1.0, 1.0, 1.0])
    #centers = np.array([[0., 0.],[0., 1.],[1., 0.],[1., 1.]])
    #widths = np.array([1.,0.01,1.,1.])
    num_objs = 2
    weights = 2*np.random.rand(num_objs)
    widths = np.random.rand(num_objs)
    centers = np.random.rand(num_objs,2)


    rbf = RbfReward(centers, weights, widths)
    print(rbf.get_reward(x))
    #rbf.get_heat_map()
    rbf.plot_heat_map()

    best_x, reward = rbf.estimate_best_placement()
    plt.plot(best_x[0], best_x[1],'*',markersize=30)
    print("best x", best_x, "best reward", reward)

    #plot centers of objects
    for c in centers:
        plt.plot(c[0],c[1],'o',markersize=20)
    #plt.show()

    plt.show()


#used to plot between example
def plot_nextto_reward():


    #assumes grid in [0,1]x[0,1] region

    x = np.array([0., 0.])
    #weights = np.array([1.0, 1.0, 1.0, 1.0])
    #centers = np.array([[0., 0.],[0., 1.],[1., 0.],[1., 1.]])
    #widths = np.array([1.,0.01,1.,1.])

    w1 = 2.0
    w2 = -1.0
    weights = np.array([w1,w2])
    width1 = 0.6
    width2 = 0.05
    widths = np.array([width1, width2])
    c1 = [0.5,0.5]
    centers = np.array([c1,c1])


    rbf = RbfReward(centers, weights, widths)
    print(rbf.get_reward(x))
    #rbf.get_heat_map()
    rbf.plot_heat_map()

    best_x, reward = rbf.estimate_best_placement()
    plt.plot(best_x[0], best_x[1],'*',markersize=30)
    print("best x", best_x, "best reward", reward)

    #plot centers of objects
    for c in centers:
        plt.plot(c[0],c[1],'bo',markersize=20)
    #plt.show()

    plt.show()

if __name__=="__main__":
    plot_between_reward_positive()
    #plot_nextto_reward()
