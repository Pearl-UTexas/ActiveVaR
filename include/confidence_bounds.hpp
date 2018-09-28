#ifndef confidence_bounds_hpp
#define confidence_bounds_hpp
#include "mdp.hpp"
#include <math.h>
#include <string>
#include <unordered_map>

double evaluateExpectedReturn(  vector<unsigned int> & policy, 
		MDP* evalMDP, double eps);


void policyValueIteration(  vector<unsigned int> & policy, 
		MDP* evalMDP, double eps, double* V);

double getExpectedReturn(  MDP* mdp);
double getAverageValueFromStartStates(double* V, bool* init, unsigned int numStates);
double** calculateStateExpectedFeatureCounts(vector<unsigned int> & policy, FeatureGridMDP* fmdp, double eps);
double* calculateExpectedFeatureCounts(vector<unsigned int> & policy, FeatureGridMDP* fmdp, double eps);
double* calculateEmpiricalExpectedFeatureCounts(vector<vector<pair<unsigned int,unsigned int> > > trajectories, FeatureGridMDP* fmdp);

double evaluateExpectedReturn(  vector<unsigned int> & policy, 
		MDP* evalMDP, double eps)
{
	//initialize values to zero
	unsigned int numStates = evalMDP->getNumStates();
	double V[numStates];
	for(unsigned int i=0; i<numStates; i++) V[i] = 0.0;

	//get value of policy in evalMDP
	policyValueIteration(policy, evalMDP, eps, V);

	//get expected value of policy evaluated on evalMDP over starting dist
	bool* init = evalMDP->getInitialStates();
	return getAverageValueFromStartStates(V, init, numStates);
}


vector<double> evaluateExpectedReturnVector(  vector<unsigned int> & policy, 
		MDP* evalMDP, double eps)
{
	//initialize values to zero
	vector<double> init_state_returns;
	unsigned int numStates = evalMDP->getNumStates();
	double V[numStates];
	for(unsigned int i=0; i<numStates; i++) V[i] = 0.0;

	//get value of policy in evalMDP
	policyValueIteration(policy, evalMDP, eps, V);

	//get expected value of policy evaluated on evalMDP over starting dist
	bool* init = evalMDP->getInitialStates();
	//check if there is at least one starting state
	bool startStateExists = false;
	for(unsigned int i=0; i<numStates; i++)
		if(init[i])
			startStateExists = true;
	assert(startStateExists);

	for(unsigned int s=0; s < numStates; s++)
	{
		if(init[s])
		{
			init_state_returns.push_back(V[s]);
		}
	}
	return init_state_returns;

}




double getAverageValueFromStartStates(double* V, bool* init, unsigned int numStates)
{
	//check if there is at least one starting state
	bool startStateExists = false;
	for(unsigned int i=0; i<numStates; i++)
		if(init[i])
			startStateExists = true;
	assert(startStateExists);
	double valSum = 0;
	int initCount = 0;
	for(unsigned int s=0; s < numStates; s++)
	{
		if(init[s])
		{
			valSum += V[s];
			initCount++;
		}
	}
	return valSum / initCount;
}

//Updates vector of values V to be value of using policy in evalMDP
//run value iteration until convergence using policy actions rather than argmax
void policyValueIteration(  vector<unsigned int> & policy, 
		MDP* evalMDP, double eps, double* V)
{
	double delta;
	double discount = evalMDP->getDiscount();
	double*** T = evalMDP->getTransitions();
	//repeat until convergence within error eps
	do
	{
		unsigned int numStates = evalMDP->getNumStates();

		//cout << "--------" << endl;
		//displayAsGrid(V);
		delta = 0;
		//update value of each state
		// cout << eps * (1 - discount) / discount << "," << delta << endl;

		for(unsigned int s1 = 0; s1 < numStates; s1++)
		{
			double tempV = 0;
			//add reward
			tempV += evalMDP->getReward(s1);
			//add discounted value of next state based on policy action
			int policy_action = policy[s1];
			//calculate expected utility of taking action a in state s1
			double expUtil = 0;

			for(unsigned int s2 = 0; s2 < numStates; s2++)
			{
				expUtil += T[s1][policy_action][s2] * V[s2];
			}
			tempV += discount * expUtil;

			//update delta to track convergence
			double absDiff = abs(tempV - V[s1]);
			if(absDiff > delta)
				delta = absDiff;
			V[s1] = tempV;
		}

	}
	while(delta > eps);

}

//returns the expected return of the optimal policy for the input mdp
//assumes value iteration has already been run
double getExpectedReturn(  MDP* mdp)
{
	unsigned int numStates = mdp->getNumStates();
	double* V = mdp->getValues();
	bool* init = mdp->getInitialStates();
	return getAverageValueFromStartStates(V, init, numStates);

}

//returns the expected return of the optimal policy for the input mdp
//assumes value iteration has already been run
vector<double> getExpectedReturnVector(  MDP* mdp)
{
	vector<double> init_state_returns;
	unsigned int numStates = mdp->getNumStates();
	double* V = mdp->getValues();
	bool* init = mdp->getInitialStates();

	//check if there is at least one starting state
	bool startStateExists = false;
	for(unsigned int i=0; i<numStates; i++)
		if(init[i])
			startStateExists = true;
	assert(startStateExists);

	for(unsigned int s=0; s < numStates; s++)
	{
		if(init[s])
		{
			init_state_returns.push_back(V[s]);
		}
	}
	return init_state_returns;
}

//uses an analogue to policy evaluation to calculate the expected features for each state
//runs until change is less than eps
double** calculateStateExpectedFeatureCounts(vector<unsigned int> & policy, FeatureGridMDP* fmdp, double eps)
{
	unsigned int numStates = fmdp->getNumStates();
	unsigned int numFeatures = fmdp->getNumFeatures();
	double** stateFeatures = fmdp->getStateFeatures();
	double discount = fmdp->getDiscount();
	double*** T = fmdp->getTransitions();


	//initalize 2-d array for storing feature weights
	double** featureCounts = new double*[numStates];
	for(unsigned int s = 0; s < numStates; s++)
		featureCounts[s] = new double[numFeatures];
	for(unsigned int s = 0; s < numStates; s++)
		for(unsigned int f = 0; f < numFeatures; f++)
			featureCounts[s][f] = 0;

	//run feature count iteration
	double delta;

	//repeat until convergence within error eps
	do
	{
		delta = 0;
		for(unsigned int s1 = 0; s1 < numStates; s1++)
		{
			//cout << "for state: " << s1 << endl;
			//use temp array to store accumulated, discounted feature counts
			double tempF[numFeatures];
			for(unsigned int f = 0; f < numFeatures; f++)
				tempF[f] = 0;            

			//add current state features
			for(unsigned int f =0; f < numFeatures; f++)
				tempF[f] += stateFeatures[s1][f];

			//update value of each reachable next state following policy
			unsigned int policyAction = policy[s1];
			double transitionFeatures[numFeatures];
			for(unsigned int f = 0; f < numFeatures; f++)
				transitionFeatures[f] = 0;

			for(unsigned int s2 = 0; s2 < numStates; s2++)
			{
				if(T[s1][policyAction][s2] > 0)
				{       
					//cout << "adding transition to state: " << s2 << endl;
					//accumulate features for state s2
					for(unsigned int f = 0; f < numFeatures; f++)
						transitionFeatures[f] += T[s1][policyAction][s2] * featureCounts[s2][f];
				}
			}
			//add discounted transition features to tempF
			for(unsigned int f = 0; f < numFeatures; f++)
			{
				tempF[f] += discount * transitionFeatures[f];
				//update delta to track convergence
				double absDiff = abs(tempF[f] - featureCounts[s1][f]);
				if(absDiff > delta)
					delta = absDiff;
				featureCounts[s1][f] = tempF[f];
			}
		}
	}
	while(delta > eps);

	return  featureCounts;
}


//uses an analogue to policy evaluation to calculate the expected features for each state
//runs until change is less than eps
double** calculateStateExpectedFeatureCounts(vector<vector<double> > & policy, FeatureGridMDP* fmdp, double eps)
{
	unsigned int numStates = fmdp->getNumStates();
	unsigned int numActions = fmdp->getNumActions();
	unsigned int numFeatures = fmdp->getNumFeatures();
	double** stateFeatures = fmdp->getStateFeatures();
	double discount = fmdp->getDiscount();
	double*** T = fmdp->getTransitions();


	//initalize 2-d array for storing feature weights
	double** featureCounts = new double*[numStates];
	for(unsigned int s = 0; s < numStates; s++)
		featureCounts[s] = new double[numFeatures];
	for(unsigned int s = 0; s < numStates; s++)
		for(unsigned int f = 0; f < numFeatures; f++)
			featureCounts[s][f] = 0;

	//run feature count iteration
	double delta;

	//repeat until convergence within error eps
	do
	{
		delta = 0;
		//update value of each state

		for(unsigned int s1 = 0; s1 < numStates; s1++)
		{
			//use temp array to store accumulated, discounted feature counts
			double tempF[numFeatures];
			for(unsigned int f = 0; f < numFeatures; f++)
				tempF[f] = 0;            

			//add current state features
			for(unsigned int f =0; f < numFeatures; f++)
				tempF[f] += stateFeatures[s1][f];

			//update value of each reachable next state following policy
			double transitionFeatures[numFeatures];
			for(unsigned int f = 0; f < numFeatures; f++)
				transitionFeatures[f] = 0;

			for(unsigned int s2 = 0; s2 < numStates; s2++)
			{
				for(unsigned int a = 0; a < numActions; a++)
				{
					if(T[s1][a][s2] > 0 && policy[s1][a] > 0)
					{       
						//accumulate features for state s2
						for(unsigned int f = 0; f < numFeatures; f++)
							transitionFeatures[f] += policy[s1][a] * T[s1][a][s2] * featureCounts[s2][f];
					}
				}
			}
			//add discounted transition features to tempF
			for(unsigned int f = 0; f < numFeatures; f++)
			{
				tempF[f] += discount * transitionFeatures[f];
				//update delta to track convergence
				double absDiff = abs(tempF[f] - featureCounts[s1][f]);
				if(absDiff > delta)
					delta = absDiff;
				featureCounts[s1][f] = tempF[f];
			}
		}
	}
	while(delta > eps);

	return  featureCounts;
}



double* calculateExpectedFeatureCounts(vector<unsigned int> & policy, FeatureGridMDP* fmdp, double eps)
{
	//average over initial state distribution (assumes all initial states equally likely)
	double** stateFcounts = calculateStateExpectedFeatureCounts(policy, fmdp, eps);
	unsigned int numStates = fmdp -> getNumStates();
	unsigned int numFeatures = fmdp -> getNumFeatures();
	int numInitialStates = 0;

	double* expFeatureCounts = new double[numFeatures];
	fill(expFeatureCounts, expFeatureCounts + numFeatures, 0);

	for(unsigned int s = 0; s < numStates; s++)
		if(fmdp -> isInitialState(s))
		{
			numInitialStates++;
			for(unsigned int f = 0; f < numFeatures; f++)
				expFeatureCounts[f] += stateFcounts[s][f];
		}

	//divide by number of initial states
	for(unsigned int f = 0; f < numFeatures; f++)
		expFeatureCounts[f] /= numInitialStates;

	//clean up
	for(unsigned int s = 0; s < numStates; s++)
		delete[] stateFcounts[s];
	delete[] stateFcounts;    

	return expFeatureCounts;

}

double* calculateEmpiricalExpectedFeatureCounts(vector<vector<pair<unsigned int,unsigned int> > > trajectories, FeatureGridMDP* fmdp)
{
	unsigned int numFeatures = fmdp->getNumFeatures();
	double gamma = fmdp->getDiscount();
	double** stateFeatures = fmdp->getStateFeatures();

	//average over all trajectories the discounted feature weights
	double* aveFeatureCounts = new double[numFeatures];
	fill(aveFeatureCounts, aveFeatureCounts + numFeatures, 0);
	for(vector<pair<unsigned int, unsigned int> > traj : trajectories)
	{
		for(unsigned int t = 0; t < traj.size(); t++)
		{
			pair<unsigned int, unsigned int> sa = traj[t];
			unsigned int state = sa.first;
			for(unsigned int f = 0; f < numFeatures; f++)
				aveFeatureCounts[f] += pow(gamma, t) * stateFeatures[state][f];
		}
	}
	//divide by number of demos
	for(unsigned int f = 0; f < numFeatures; f++)
		aveFeatureCounts[f] /= trajectories.size();
	return aveFeatureCounts;
}




//calculate based on demos and policy and take infintity norm of difference
double calculateWorstCaseFeatureCountBound(vector<unsigned int> & policy, FeatureGridMDP* fmdp, vector<vector<pair<unsigned int,unsigned int> > > trajectories, double eps)
{
	unsigned int numFeatures = fmdp -> getNumFeatures();
	double* muhat_star = calculateEmpiricalExpectedFeatureCounts(trajectories,
			fmdp);    
	double* mu_pieval = calculateExpectedFeatureCounts(policy, fmdp, eps);
	//calculate the infinity norm of the difference
	double maxAbsDiff = 0;
	for(unsigned int f = 0; f < numFeatures; f++)
	{
		double absDiff = abs(muhat_star[f] - mu_pieval[f]);
		if(absDiff > maxAbsDiff)
			maxAbsDiff = absDiff;
	}
	//clean up
	delete[] muhat_star;
	delete[] mu_pieval;     
	return maxAbsDiff;
}


#endif
