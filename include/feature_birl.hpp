
#ifndef feature_birl_h
#define feature_birl_h

#include <cmath>
#include <stdlib.h>
#include <vector>
#include <numeric>
#include <math.h>
#include "mdp.hpp"

#include "../include/unit_norm_sampling.hpp"

using namespace std;

class FeatureBIRL { // BIRL process

	protected:

		double r_min, r_max, step_size;
		unsigned int chain_length;
		unsigned int grid_height, grid_width;
		double alpha;
		unsigned int iteration;
		const unsigned int skip = 5;

		void initializeMDP();
		vector<pair<unsigned int,unsigned int> > positive_demonstration;
		vector<pair<unsigned int,unsigned int> > negative_demonstration;

		void modifyFeatureWeightsRandomly(FeatureGridMDP * gmdp, double step_size, int count);

		double* posteriors = nullptr;
		FeatureGridMDP* MAPmdp = nullptr;
		FeatureGridMDP* AVGmdp = nullptr;
		double MAPposterior;
		//unsigned int sample_length;
		unsigned int * frequency;
		FeatureGridMDP** R_chain_long = nullptr;
		double *  posteriors_long = nullptr;
		bool stochastic=false;

	public:

		FeatureGridMDP* mdp = nullptr; //original MDP 
		FeatureGridMDP** R_chain = nullptr; //storing the rewards along the way

		~FeatureBIRL(){
			if(R_chain != nullptr) {
				for(unsigned int i=0; i<chain_length; i++) if(R_chain[i] != nullptr) delete R_chain[i];
				delete []R_chain;
			}
			if(posteriors != nullptr) delete []posteriors;
			delete MAPmdp;
			delete AVGmdp;
			delete mdp; 
			delete [] R_chain_long;
			delete [] posteriors_long;

		}


		FeatureBIRL(FeatureGridMDP* init_mdp, double min_reward, double max_reward, unsigned int chain_len, double step, double conf, bool stochastic=false):  r_min(min_reward), r_max(max_reward), step_size(step), chain_length(chain_len), alpha(conf), stochastic(stochastic) { 

			unsigned int grid_height = init_mdp -> getGridHeight();
			unsigned int grid_width = init_mdp -> getGridWidth();
			bool* initStates = init_mdp -> getInitialStates();
			bool* termStates = init_mdp -> getTerminalStates();
			unsigned int nfeatures = init_mdp -> getNumFeatures();
			double* fweights = init_mdp -> getFeatureWeights();
			double** sfeatures = init_mdp -> getStateFeatures();
			double gamma = init_mdp -> getDiscount();

			//copy init_mdp
			mdp = new FeatureGridMDP(grid_width, grid_height, initStates, termStates, nfeatures, fweights, sfeatures, stochastic, gamma);


			MAPmdp = new FeatureGridMDP(grid_width, grid_height, initStates, termStates, nfeatures, fweights, sfeatures, stochastic, gamma);
			AVGmdp = new FeatureGridMDP(grid_width, grid_height, initStates, termStates, nfeatures, fweights, sfeatures, stochastic, gamma);

			MAPmdp->setFeatureWeights(mdp->getFeatureWeights());
			AVGmdp->setFeatureWeights(mdp->getFeatureWeights());
			MAPposterior = 0;

			R_chain = new FeatureGridMDP*[chain_length];
			posteriors = new double[chain_length];   
			R_chain_long = new FeatureGridMDP*[chain_length*skip];
			posteriors_long = new double[chain_length*skip];

			iteration = 0;

		}; 

		FeatureGridMDP* getMAPmdp(){return MAPmdp;}
		FeatureGridMDP* getAVGmdp(){return AVGmdp;}

		double getMAPposterior(){return MAPposterior;}

		void addPositiveDemo(pair<unsigned int,unsigned int> demo) { positive_demonstration.push_back(demo); }; 
		void addNegativeDemo(pair<unsigned int,unsigned int> demo) { negative_demonstration.push_back(demo); };
		void addPositiveDemos(vector<pair<unsigned int,unsigned int> > demos);
		void addNegativeDemos(vector<pair<unsigned int,unsigned int> > demos);
		void run();
		void displayPositiveDemos();
		void displayNegativeDemos();
		void displayDemos();
		double getMinReward(){return r_min;};
		double getMaxReward(){return r_max;};
		double getStepSize(){return step_size;};
		double getAlpha(){return alpha;}
		void setAlpha(double a){alpha = a;};
		unsigned int getChainLength(){return chain_length;};
		vector<pair<unsigned int,unsigned int> >& getPositiveDemos(){ return positive_demonstration; };
		vector<pair<unsigned int,unsigned int> >& getNegativeDemos(){ return negative_demonstration; };
		unsigned int getNumDemonstrations(){ return positive_demonstration.size() + negative_demonstration.size(); };
		FeatureGridMDP** getRewardChain(){ return R_chain; };
		double* getPosteriorChain(){ return posteriors; };
		FeatureGridMDP* getMDP(){ return mdp;};
		double calculatePosterior(FeatureGridMDP* gmdp);
		double logsumexp(double* nums, unsigned int size);
		bool isDemonstration(pair<double,double> s_a);
		void removeDemonstration(pair<unsigned int,unsigned int> s_a);
		void removeAllDemostrations();
		void manifoldL1UnitBallWalkAllSteps(FeatureGridMDP * gmdp, double step);

};

void FeatureBIRL::removeDemonstration(pair<unsigned int,unsigned int> s_a)
{
	positive_demonstration.erase(std::remove(positive_demonstration.begin(), positive_demonstration.end(), s_a), positive_demonstration.end());
	negative_demonstration.erase(std::remove(negative_demonstration.begin(), negative_demonstration.end(), s_a), negative_demonstration.end());
}

void FeatureBIRL::run()
{

	cout.precision(10);

	//clear out previous values if they exist
	if(iteration > 0) 
		for(unsigned int i=0; i< chain_length; i++) 
		{
			if(R_chain[i] != nullptr){
				delete R_chain[i];  
				R_chain[i] = nullptr;
			}
		}
	iteration++;

	initializeMDP(); 
	mdp->valueIteration(0.001);
	double posterior = calculatePosterior(mdp);
	double new_posterior;

	R_chain_long[0] = mdp->deepcopy(); 
	posteriors_long[0] = posterior; 

	int reject_cnt = 0;
	int step_count = 5;

	//BIRL iterations 
	for(unsigned int itr=1; itr < chain_length*skip; itr++)
	{
		FeatureGridMDP* temp_mdp = mdp->deepcopy(); 
		modifyFeatureWeightsRandomly(temp_mdp, step_size, step_count);

		temp_mdp->valueIteration(0.001);
		new_posterior = calculatePosterior(temp_mdp);

		double probability = min((double)1.0, exp(new_posterior - posterior));

		//transition with probability
		double r = ((double) rand() / (RAND_MAX));

		if ( r < probability ) 
		{
			delete mdp;
			mdp = temp_mdp->deepcopy(); 
			posterior = new_posterior;
			R_chain_long[itr] = temp_mdp;
			posteriors_long[itr] = new_posterior; 
		}else {
			reject_cnt++;
			R_chain_long[itr] = mdp->deepcopy();
			posteriors_long[itr] = posteriors_long[itr-1];  
			delete temp_mdp;
		}         
	}

	MAPposterior = posteriors_long[0];
	delete MAPmdp;
	MAPmdp = mdp->deepcopy(); 

	for( unsigned int i = 0; i < chain_length*skip; i++)
	{
		if(i % skip == skip - 1)
		{
			unsigned int itr = i/skip;
			R_chain[itr] = R_chain_long[i];
			posteriors[itr] = posteriors_long[i]; 
			if(posteriors[itr] > MAPposterior)
			{
				MAPposterior = posteriors[itr];
				delete MAPmdp;
				MAPmdp = R_chain_long[i]->deepcopy();
			}

		}
		else delete R_chain_long[i];
	}

	MAPmdp->computeCachedRewards();
	MAPmdp->calculateQValues();
}


double FeatureBIRL::logsumexp(double* nums, unsigned int size) {
	double max_exp = nums[0];
	double sum = 0.0;
	unsigned int i;

	for (i = 1 ; i < size ; i++)
	{
		if (nums[i] > max_exp)
			max_exp = nums[i];
	}

	for (i = 0; i < size ; i++)
		sum += exp(nums[i] - max_exp);

	return log(sum) + max_exp;
}

double FeatureBIRL::calculatePosterior(FeatureGridMDP* gmdp) //assuming uniform prior
{

	double posterior = 0;
	unsigned int state, action;
	unsigned int numActions = gmdp->getNumActions();
	gmdp->calculateQValues();

	// "-- Positive Demos --" 
	for(unsigned int i=0; i < positive_demonstration.size(); i++)
	{
		pair<unsigned int,unsigned int> demo = positive_demonstration[i];
		state =  demo.first;
		action = demo.second; 

		double Z [numActions]; //

		for(unsigned int a = 0; a < numActions; a++) Z[a] = alpha*(gmdp->getQValue(state,a));
		posterior += alpha*(gmdp->getQValue(state,action)) - logsumexp(Z, numActions);
	}

	// "-- Negative Demos --" 
	for(unsigned int i=0; i < negative_demonstration.size(); i++)
	{
		pair<unsigned int,unsigned int> demo = negative_demonstration[i];
		state =  demo.first;
		action = demo.second;
		double Z [numActions]; //
		for(unsigned int a = 0; a < numActions; a++)  Z[a] = alpha*(gmdp->getQValue(state,a));

		unsigned int ct = 0;
		double Z2 [numActions - 1]; 
		for(unsigned int a = 0; a < numActions; a++) 
		{
			if(a != action) Z2[ct++] = alpha*(gmdp->getQValue(state,a));
		}

		posterior += logsumexp(Z2, numActions-1) - logsumexp(Z, numActions);
	}
	return posterior;
}

void FeatureBIRL::modifyFeatureWeightsRandomly(FeatureGridMDP * gmdp, double step, int count)
{
	unsigned int state; 
	double change;
	double weight;

	while(count >0)
	{
		state = rand() % gmdp->getNumFeatures();
		change  = pow(-1,rand()%2)*step;
		weight = max(min(gmdp->getWeight(state) + change, r_max), r_min);
		gmdp->setFeatureWeight(state, weight);
		count--;
	}
	//normalize feature weights
	double total_weight = 0;
	for(unsigned int idx = 0; idx < gmdp->getNumFeatures(); idx++) total_weight += abs(gmdp->getWeight(idx));
	for(unsigned int idx = 0; idx < gmdp->getNumFeatures(); idx++) gmdp->setFeatureWeight(idx,gmdp->getWeight(idx)/total_weight);

}

void FeatureBIRL::removeAllDemostrations()
{
	positive_demonstration.clear();
	negative_demonstration.clear();
}

void FeatureBIRL::addPositiveDemos(vector<pair<unsigned int,unsigned int> > demos)
{
	for(unsigned int i=0; i < demos.size(); i++)  positive_demonstration.push_back(demos[i]);
}
void FeatureBIRL::addNegativeDemos(vector<pair<unsigned int,unsigned int> > demos)
{
	for(unsigned int i=0; i < demos.size(); i++)  negative_demonstration.push_back(demos[i]);
}

void FeatureBIRL::displayDemos()
{
	displayPositiveDemos();
	displayNegativeDemos();
}

void FeatureBIRL::displayPositiveDemos()
{
	if(positive_demonstration.size() !=0 ) cout << "\n-- Positive Demos --" << endl;
	for(unsigned int i=0; i < positive_demonstration.size(); i++)
	{
		pair<unsigned int,unsigned int> demo = positive_demonstration[i];
		cout << " (" << demo.first << "," << demo.second << "), "; 

	}
	cout << endl;
}
void FeatureBIRL::displayNegativeDemos()
{
	if(negative_demonstration.size() != 0) cout << "\n-- Negative Demos --" << endl;
	for(unsigned int i=0; i < negative_demonstration.size(); i++)
	{
		pair<unsigned int,unsigned int> demo = negative_demonstration[i];
		cout << " (" << demo.first << "," << demo.second << "), "; 

	}
	cout << endl;
}

void FeatureBIRL::initializeMDP()
{
	double* weights = new double[mdp->getNumFeatures()];
	for(unsigned int s=0; s<mdp->getNumFeatures(); s++)
	{
		weights[s] = - float(rand()%100)/100;
	}
	//normalize feature weights
	double total_weight = 0;
	for(unsigned int idx = 0; idx < mdp->getNumFeatures(); idx++) total_weight += abs(weights[idx]);
	for(unsigned int idx = 0; idx < mdp->getNumFeatures(); idx++) weights[idx] /= total_weight;

	mdp->setFeatureWeights(weights);
	delete []weights;

}

bool FeatureBIRL::isDemonstration(pair<double,double> s_a)
{
	for(unsigned int i=0; i < positive_demonstration.size(); i++)
	{
		if(positive_demonstration[i].first == s_a.first && positive_demonstration[i].second == s_a.second) return true;
	}
	for(unsigned int i=0; i < negative_demonstration.size(); i++)
	{
		if(negative_demonstration[i].first == s_a.first && negative_demonstration[i].second == s_a.second) return true;
	}
	return false;

}

void FeatureBIRL::manifoldL1UnitBallWalkAllSteps(FeatureGridMDP * gmdp, double step)
{
	unsigned int numFeatures = gmdp->getNumFeatures();
	double* newWeights = take_all_manifold_l1_steps(gmdp->getFeatureWeights(), numFeatures, step);
	gmdp->setFeatureWeights(newWeights);
	delete [] newWeights;
}


#endif

