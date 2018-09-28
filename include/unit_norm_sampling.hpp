#ifndef unit_norm_sampling_hpp
#define unit_norm_sampling_hpp

#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdlib.h>
#include <ctime>
#include <assert.h>
#include <string>

using namespace std;

double* sample_unit_L1_norm(int dim);
double* updown_l1_norm_walk(const double* weights, int dim, double step);
bool isEqual(double x, double y);
double l1_norm(double* vals, int num_vals);
pair<double,double> manifold_l1_step(double cur_val1, double cur_val2, string rand_dir, double rand_step);
double* random_manifold_l1_step(const double* weights, int dim, double step_size, int num_steps);
vector<pair<int,int> > generateAllUniquePairs(int* values, int numValues);
void permuteRandomly(int* values, int numValues);
double* take_all_manifold_l1_steps(const double* weights, int dim, double step_size);



double l1_norm(double* vals, int num_vals)
{
    double abs_sum = 0;
    for(int i=0; i<num_vals; i++)
        abs_sum += abs(vals[i]);
    return abs_sum;

}

//Sample random uniformly from L1 unit ball of dimension dim
double* sample_unit_L1_norm(int dim)
{
    //sample dim numbers from exp(-|x|)/2
    double* sample = new double[dim];
    double abs_sum = 0;
    for(int d=0; d<dim; d++)
    {
        //draw random number in [0,1]
        double z = rand() / (double) RAND_MAX;
        if(z < 0.5)
            sample[d] = log(2.0 * z);
        else
            sample[d] = -log(2.0 - 2.0 * z);
        //keep track of sum
        abs_sum += abs(sample[d]);
    }
    //normalize samples
    for(int d=0; d<dim; d++)
        sample[d] /= abs_sum;
    return sample;
}

//pick one dimension to go up and another one to go down in magnitude.
double* updown_l1_norm_walk(const double* weights, int dim, double step)
{

    //copy weights
    double* sample = new double[dim];
    for(int d = 0; d < dim; d++)
        sample[d] = weights[d];
    //pick random dimensions making sure not to pick two zeros
    int rand_dim1;
    int rand_dim2;
    do
    {
        rand_dim1 = rand() % dim;
        rand_dim2 = rand() % dim;
        while(rand_dim2 == rand_dim1)
            rand_dim2 = rand() % dim;
    }
    while(isEqual(weights[rand_dim1], 0.0) && isEqual(weights[rand_dim2], 0.0));
    //both are not 0.0 so if one is 0.0 the other is 1.0
    //if one dimension has zero value increase magnitude and decrease other (mag = 1.0)
    int rand_dir1 = 0;;
    int rand_dir2 = 0;
    if(isEqual(weights[rand_dim1], 0.0))
    {
        //pick random direction to increase magnitude (either positive or negative)
        rand_dir1 = 2 * (rand() % 2) - 1;
        //other dimension should decrease in magnitude
        rand_dir2 = -1;
    }
    else if(isEqual(weights[rand_dim2], 0.0))
    {
        //pick random direction to increase magnitude
        rand_dir2 = 2 * (rand() % 2) - 1;
        //other dimension should decrease in magnitude (since it is 1.0 in abs val)
        rand_dir1 = -1;
    }
    //otherwise just have one go up and the other go down
    else
    {
        //pick to go up or down in magnitude for dim1
        rand_dir1 = 2 * (rand() % 2) - 1;
        rand_dir2 = -rand_dir1;
    }
    //update weights appropriately taking into account positive and negatives
    if(weights[rand_dim1] >= 0)
        sample[rand_dim1] += rand_dir1 * step;
    else 
        sample[rand_dim1] += (-1) * rand_dir1 * step;
    
    if(weights[rand_dim2] >= 0)
        sample[rand_dim2] += rand_dir2 * step;
    else 
        sample[rand_dim2] += (-1) * rand_dir2 * step;
    
    return sample;
}

bool isEqual(double x, double y)
{
    double eps = 1e-4;
    if(abs(x-y) < eps)
        return true;
    else
        return false;

}



pair<double,double> manifold_l1_step(double cur_val1, double cur_val2, string rand_dir, double rand_step)
{
    double slack = abs(cur_val1) + abs(cur_val2);
    string clockwise_pos[] = {"++","+-","--","-+"};
    int clockwise_dir[] = {+1,-1,+1,-1}; //only need direction of the first dim
    string counterclockwise_pos[] = {"++","-+","--","+-"};
    int counterclockwise_dir[] = {-1,+1,-1,+1};
    
    //find current quadrant
    bool sign1 = (cur_val1 >= 0);
    bool sign2 = (cur_val2 >= 0);
    
    string cycle_pos;
    if(sign1 && sign2)
        cycle_pos = "++";
    else if(sign1 && !sign2)
        cycle_pos = "+-";
    else if(!sign1 && sign2)
        cycle_pos = "-+";
    else if(!sign1 && !sign2)
        cycle_pos = "--";
        
        
    //find direction up or down
    int cycle_indx = 0;
    if(rand_dir == "clockwise")
    {
        for(int i=0; i<4; i++)
        {
            if(clockwise_pos[i] == cycle_pos)
            {
                cycle_indx = i;
                break;
            }
        }
    }
    else
    {
       for(int i=0; i<4; i++)
        {
            if(counterclockwise_pos[i] == cycle_pos)
            {
                cycle_indx = i;
                break;
            }
        }
    }
    int cycle_dir;
    double step_remaining = rand_step;
    while(!isEqual(step_remaining,0.0))
    {    
        //find direction up or down
        if(rand_dir == "clockwise")
            cycle_dir = clockwise_dir[cycle_indx];
        else
            cycle_dir = counterclockwise_dir[cycle_indx];
        
        double max_step = step_remaining;
        // check if over/underflow based on cycle_dir
        if((cycle_dir == 1) && ((abs(cur_val1) + cycle_dir * step_remaining) > slack))
        { 
            //take max step and update index based on direction
            max_step = slack - abs(cur_val1);
            cycle_indx = (cycle_indx + 1) % 4;
        }
        else if( (cycle_dir == -1) && ((abs(cur_val1) + cycle_dir * step_remaining) < 0.0))
        {
            max_step = abs(cur_val1);
            cycle_indx = (cycle_indx + 1) % 4;
        }
        cur_val1 = abs(cur_val1) + cycle_dir * max_step;
        cur_val2 = abs(cur_val2) - cycle_dir * max_step;
        step_remaining -= max_step;
        
    }    
    //multiply by signs to get in correct quadrant
    if(rand_dir == "clockwise")
        cycle_pos = clockwise_pos[cycle_indx];
    else
        cycle_pos = counterclockwise_pos[cycle_indx];
    if(cycle_pos == "-+")
        cur_val1 *= -1;
    else if(cycle_pos == "+-")
        cur_val2*= -1;
    else if(cycle_pos == "--")
    {
        cur_val1 *= -1;
        cur_val2 *= -1;
    }
    assert(isEqual(abs(cur_val1) + abs(cur_val2), slack));
    return make_pair(cur_val1, cur_val2);
}

double* random_manifold_l1_step(const double* weights, int dim, double step_size, int num_steps)
{
    //copy weights
    double* sample = new double[dim];
    for(int d = 0; d < dim; d++)
        sample[d] = weights[d];
    
    for(int s = 0; s < num_steps; s++)
    {
        //pick random direction clockwise or counterclockwise
        string dir;
        if(rand() % 2 == 0)
            dir = "clockwise";
        else
            dir = "counterclockwise";
        //pick two random dimensions to vary along 1-d manifold
        // but don't pick two zeros since there is no slack there.
        int rand_dim1;
        int rand_dim2;
        do
        {
            rand_dim1 = rand() % dim;
            rand_dim2 = rand() % dim;
            while(rand_dim2 == rand_dim1)
                rand_dim2 = rand() % dim;
        }
        while(isEqual(sample[rand_dim1], 0.0) && isEqual(sample[rand_dim2], 0.0));
        //take step along two chosen dimensions
        pair<double,double> new_vals = manifold_l1_step(sample[rand_dim1], sample[rand_dim2], dir, step_size);
        //update vals
        sample[rand_dim1] = new_vals.first;
        sample[rand_dim2] = new_vals.second;
    }    
    return sample;
 
}

//randomly shuffle elements
void permuteRandomly(int* values, int numValues)
{
    int unsorted = numValues-1;
    while(unsorted > 0)
    {
        //pick random number in [0, unsorted) 
        int randIndex = rand() % unsorted;
        //send value at random index to unsorted index and decrement unsorted
        int temp = values[randIndex];
        values[randIndex] = values[unsorted];
        values[unsorted] = temp;
        unsorted--;
    }    
    
}


vector<pair<int,int> > generateAllUniquePairs(int* values, int numValues)
{
    //calculate numValues choose 2
    vector<pair<int, int> > pairs;
    for(int i = 0; i < numValues; i++)
        for(int j = i+1; j < numValues; j++)
            pairs.push_back(make_pair(values[i],values[j]));
    return pairs;

}

double* take_all_manifold_l1_steps(const double* weights, int dim, double step_size)
{
    //copy weights
    double* sample = new double[dim];
    for(int d = 0; d < dim; d++)
        sample[d] = weights[d];
    
    //for each pairing of dimensions, take a step if both not zero
    int numFeatures = dim;
    int permutation[numFeatures];
    for(int i=0; i<numFeatures; i++)
        permutation[i] = i;
        
    //randomly select pairings of dimensions
    permuteRandomly(permutation, numFeatures);   
    vector<pair<int, int> > tuples = generateAllUniquePairs(permutation, numFeatures);
    for(pair<int,int> t : tuples)
    {
        //pick random direction clockwise or counterclockwise
        string dir;
        if(rand() % 2 == 0)
            dir = "clockwise";
        else
            dir = "counterclockwise";

        //step in direction along manifold
        int dim1 = t.first;
        int dim2 = t.second;
        if(!isEqual(sample[dim1], 0.0) || !isEqual(sample[dim2], 0.0))
        {
            pair<double,double> new_vals = manifold_l1_step(sample[dim1], sample[dim2], dir, step_size);
            //update vals
            sample[dim1] = new_vals.first;
            sample[dim2] = new_vals.second;
        }
    }
       
    return sample;
 
}



#endif
