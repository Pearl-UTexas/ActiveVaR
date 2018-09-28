
#ifndef mdp_h
#define mdp_h

#include <cstddef>
#include <cmath>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <limits>
//#define NDEBUG         //uncomment this to disable all asserts
#include <assert.h> 
#include "unit_norm_sampling.hpp"

constexpr double lowest_double = std::numeric_limits<double>::lowest();


#define NUM_ACTIONS 4

using namespace std;

double dotProduct(double x[], double y[], int length);
vector<int> argmax_all(vector<double> xs);
int roulette_wheel(vector<double> w);

class MDP { // General MDP

	protected:
		unsigned int numActions;
		unsigned int numStates;
		double discount;
		bool* initialStates = nullptr;
		bool* terminalStates = nullptr;
		bool* wallStates = nullptr;

		double* R = nullptr; 
		double*** T = nullptr;
		double* V = nullptr;
		double** Q = nullptr;
		bool Qinitialized = false;
		bool Vinitialized = false;

	public:
		MDP(double gamma, int states, int actions): numActions(actions), numStates(states), discount(gamma){
			initialStates = new bool [numStates];
			fill_n(initialStates, numStates, false);

			terminalStates = new bool [numStates];
			fill_n(terminalStates, numStates, false);

			wallStates = new bool [numStates];
			fill_n(wallStates, numStates, false);

			//initialize to all zeros
			T = new double**[numStates];
			for(unsigned int s1 = 0; s1 < numStates; s1++)
			{
				T[s1] = new double*[numActions];
				for(unsigned int a = 0; a < numActions; a++)
				{
					T[s1][a] = new double[numStates];
					for(unsigned int p = 0; p < numStates; p++) T[s1][a][p] = 0;
				}
			}
			R = new double[numStates];
			fill_n(R, numStates, 0.0);
			Q = new double*[numStates];
			V = new double[numStates];
			fill_n(V, numStates, 0.0);
			for(unsigned int s = 0; s < numStates; s++) Q[s] = new double[numActions];

		};
		void setInitialStates(bool* initials){ 
			if(initials == nullptr) return;
			for(unsigned int s = 0; s < numStates; s++) initialStates[s] = initials[s];
		};

		void setTerminalStates(bool* terminals){ 
			if(terminals == nullptr) return;
			for(unsigned int s = 0; s < numStates; s++){
				terminalStates[s] = terminals[s];
			}
		};

		void addTerminalState(unsigned int terminal){ 
			terminalStates[terminal] = true;         
		};  

		MDP* deepcopy(){
			MDP* copy = new MDP(discount, numStates, numActions);
			copy -> setInitialStates(initialStates);
			copy -> setTerminalStates(terminalStates);
			copy -> setRewards(R);
			copy -> setTransitions(T);
			return copy;
		};

		virtual ~MDP(){

			delete[] R;
			delete[] V;

			for(unsigned int s = 0; s < numStates; s++) delete[] Q[s];
			delete[] Q;

			for(unsigned int s1 = 0; s1 < numStates; s1++)
			{
				for(unsigned int a = 0; a < numActions; a++)
				{
					delete[] T[s1][a];
				}
				delete[] T[s1];
			}
			delete[] T;
			delete [] initialStates;
			delete [] terminalStates;
			delete [] wallStates;
		};

		void resetTransitions()
		{
			for(unsigned int s1 = 0; s1 < numStates; s1++)
				for(unsigned int a = 0; a < numActions; a++)
					for(unsigned int p = 0; p < numStates; p++) 
						T[s1][a][p] = 0;



		}

		double getDiscount() const{return discount;};
		void setDiscount(double gamma)
		{
			assert(gamma >=0 && gamma <=1);
			discount = gamma;
		};
		bool isQInitialized() {return Qinitialized;};
		bool isVInitialized() {return Vinitialized;};
		bool* getInitialStates() const{return initialStates;};
		bool* getTerminalStates() const{return terminalStates;};
		vector<pair<unsigned int, unsigned int>> monte_carlo_argmax_rollout(int s0, int L);
		vector<pair<unsigned int, unsigned int>> epsilon_random_rollout(int s0, int L, double epsilon);

		bool isInitialState(unsigned int s){ return initialStates[s];};
		bool isTerminalState(unsigned int s){ return terminalStates[s];};

		bool isWallState(unsigned int s){ return wallStates[s];};


		void setNumActions(unsigned int actions){ numActions = actions; };
		unsigned int getNumActions() const { return numActions; };
		void setNumStates(unsigned int states){ numStates = states; };
		unsigned int getNumStates() const{ return numStates; };

		void displayRewards(){};
		void displayPolicy(vector<unsigned int> & policy);
		void deterministicPolicyIteration(vector<unsigned int> & policy);
		void deterministicPolicyIteration(vector<unsigned int> & policy, int steps);
		void deterministicPolicyEvaluation(vector<unsigned int> & policy, int k=30); //Default argument
		void initRandDeterministicPolicy(vector<unsigned int> & policy);
		void valueIteration(double eps, double* input_v); //warm start
		void valueIteration(double eps);

		void setReward(int state, double reward){ R[state] = reward;};
		void setRewards(double*  rewards) { 
			for(unsigned int s = 0; s < numStates; s++) R[s] = rewards[s];
		};
		double* getRewards() const{ return R; };
		//virtual 
		double getReward(unsigned int state) {return R[state];}; 


		void setTransitions(double*** transitions){ T = transitions; };
		double*** getTransitions() const{ return T; };

		void calculateQValues();

		double* getValues() const{// assert(Vinitialized); 
			return V; };
		double getValue(unsigned int state){ return V[state]; };
		double getQValue(unsigned int state,unsigned int action)
		{ 
			//assert(Qinitialized); //need to call calculateQValues() first
			return Q[state][action];
		};

		double** getQValues() { 
			// assert(Qinitialized); //need to call calculateQValues() first
			return Q;
		};

		friend bool operator== (MDP & lhs, MDP & rhs);

		void setValues(double* values) {
			Vinitialized = true;
			for(unsigned int i=0;i<numStates;i++)
				V[i] = values[i];
		};
		void setQValues(double** qvalues) {
			Qinitialized = true;
			for(unsigned int s=0; s < numStates;s++)
				for(unsigned int a=0; a < numActions; a++)
					Q[s][a] = qvalues[s][a];
		}

};


class GridMDP: public MDP{ // 2D Grid MDP - defined by width*height

	private:


		enum actions {UP, DOWN, LEFT, RIGHT};

	public:
		unsigned int gridWidth;
		unsigned int gridHeight;
		bool stochasticTransitions;
		GridMDP(unsigned int width, unsigned int height, vector<unsigned int> initStates, vector<unsigned int> termStates, bool stochastic=false, double gamma=0.95): MDP(gamma,width*height,NUM_ACTIONS), gridWidth(width), gridHeight(height), stochasticTransitions(stochastic){ 
			for(unsigned int i=0; i<initStates.size(); i++)
			{
				int idx = initStates[i];
				initialStates[idx] = true;
			}
			for(unsigned int i=0; i<termStates.size(); i++)
			{
				int idx = termStates[i];
				terminalStates[idx] = true; 
			}
			if(stochastic)
				setStochasticGridTransitions();
			else
				setDeterministicGridTransitions();
		};
		GridMDP(unsigned int width, unsigned int height, bool* initStates, bool* termStates, bool stochastic=false, double gamma=0.95): MDP(gamma,width*height,NUM_ACTIONS), gridWidth(width), gridHeight(height), stochasticTransitions(stochastic){ 
			for(unsigned int i=0; i<numStates; i++)
			{
				initialStates[i] = initStates[i];
			}
			for(unsigned int i=0; i<numStates; i++)
			{
				terminalStates[i] = termStates[i]; 
			}
			if(stochastic)
				setStochasticGridTransitions();
			else
				setDeterministicGridTransitions();
		};
		void setWallState(unsigned int s){ 
			wallStates[s] = true;
			//reset all transitions
			resetTransitions();
			if(stochasticTransitions)
				setStochasticGridTransitions();
			else
				setDeterministicGridTransitions();

		};

		int getGridWidth() { return gridWidth;};
		int getGridHeight(){ return gridHeight;};
		bool isStochastic(){ return stochasticTransitions;};
		void displayRewards();
		void displayValues();
		void setDeterministicGridTransitions();
		void setStochasticGridTransitions();
		void displayTransitions();
		void displayPolicy(vector<unsigned int> & policy);
		void displayQValues();
		vector<vector<double> > getOptimalStochasticPolicy();
		void getOptimalPolicy(vector<unsigned int> & opt);
		bool isOptimalAction(unsigned int state, unsigned int action, double tolerance=1E-4);
		double getReward(unsigned int s) const;
		vector<unsigned int> getValidActions(unsigned int s){
			vector<unsigned int> actions;
			if(terminalStates[s]) return actions;
			if(s >= gridWidth) actions.push_back(UP);
			if(s < (gridHeight - 1) * gridWidth) actions.push_back(DOWN);
			if(s % gridWidth > 0) actions.push_back(LEFT);
			if(s % gridWidth < gridWidth - 1) actions.push_back(RIGHT);
			return actions;
		};
		unsigned int getNextState(unsigned int s, unsigned int a){
			if(terminalStates[s]) return s;
			if(a == UP && s >= gridWidth) return s - gridWidth;
			else if(a == DOWN && s < (gridHeight - 1) * gridWidth) return s + gridWidth;
			else if(a == LEFT && s % gridWidth > 0) return s - 1;
			else if(a == RIGHT && s % gridWidth < gridWidth - 1) return s + 1;
			else return s;
		}       


};

double GridMDP::getReward(unsigned int s) const
{ 
	assert(R != nullptr); //reward for state not defined

	return R[s];


}    

//Extension of GridMDP to allow state rewards to be linear combo of features
class FeatureGridMDP: public GridMDP{

	private:
		int numFeatures;
		double* featureWeights = nullptr;  //keeps a local copy of weights
		double** stateFeatures = nullptr;  //just a pointer to where they are defined initially...


	public:
		FeatureGridMDP(unsigned int width, unsigned int height, vector<unsigned int> initStates, vector<unsigned int> termStates, unsigned int nFeatures, double* fWeights, double** sFeatures, bool stochastic=false, double gamma=0.95): GridMDP(width, height, initStates, termStates, stochastic, gamma), numFeatures(nFeatures)
	{
		featureWeights = new double[numFeatures];
		for(int i=0; i<numFeatures; i++)
			featureWeights[i] = fWeights[i];
		stateFeatures = sFeatures;

		//compute cached rewards
		computeCachedRewards();

	};


		FeatureGridMDP(unsigned int width, unsigned int height, bool* initStates, bool* termStates, unsigned int nFeatures, double* fWeights, double** sFeatures, bool stochastic=false, double gamma=0.95): GridMDP(width, height, initStates, termStates, stochastic, gamma), numFeatures(nFeatures)
	{
		featureWeights = new double[numFeatures];
		for(int i=0; i<numFeatures; i++)
			featureWeights[i] = fWeights[i];
		stateFeatures = sFeatures;
		computeCachedRewards();

	};


		void computeCachedRewards()
		{
			for(unsigned int s = 0; s < numStates; s++) 
				R[s] = dotProduct(stateFeatures[s], featureWeights, numFeatures);
		};
		//delete featureWeights
		//stateFeatures should be deleted in main function somewhere in test script
		~FeatureGridMDP()
		{
			delete[] featureWeights;

		};

		FeatureGridMDP* deepcopy(){
			FeatureGridMDP* copy = new FeatureGridMDP(gridWidth, gridHeight, initialStates, terminalStates, numFeatures, featureWeights, stateFeatures, stochasticTransitions, discount);
			return copy;
		};
		long double L2_distance(FeatureGridMDP* in_mdp)
		{
			long double dist = 0;
			for(unsigned int i = 0; i < numStates; i++)
			{
				dist += pow(R[i] - in_mdp->R[i], 2); 
			}
			if(dist > 10000)
			{
				cout << " ????!!!!! ------> " << endl;
				displayRewards();
				in_mdp->displayRewards();
			}
			return sqrt(dist);
		}
		unsigned int getNumFeatures(){return numFeatures;};
		double* getFeatureWeights(){ return featureWeights; };
		double** getStateFeatures(){ return stateFeatures; };
		double* getStateFeature(unsigned int state){return stateFeatures[state];};
		void setFeatureWeight(unsigned int state, double weight)
		{
			featureWeights[state] = weight;
			computeCachedRewards();
		};
		//multiply features by weights for state s
		double getReward(unsigned int s) const
		{
			//use precomputed rewards
			return getCachedReward(s);
		};
		double getCachedReward(unsigned int s) const
		{
			return R[s];
		}
		void setFeatureWeights(double* fWeights)
		{
			for(int i=0; i<numFeatures; i++)
				featureWeights[i] = fWeights[i];
			computeCachedRewards();
		};
		double getWeight(unsigned int state){ return featureWeights[state]; };
		void displayFeatureWeights();
		void valueIteration(double eps);
		void valueIteration(double eps, double* input_V);
};


void FeatureGridMDP::valueIteration(double eps)
{
	Vinitialized = true;
	double delta = 0;
	for(unsigned int s = 0; s < numStates; s++)
	{
		V[s] = R[s];
	}
	double* new_V = new double[numStates];
	for(unsigned int s = 0; s < numStates; s++) new_V[s] = R[s];
	//repeat until convergence within error eps
	do
	{
		delta = 0;
		for(unsigned int s1 = 0; s1 < numStates; s1++)
		{
			new_V[s1] = R[s1];
			//add discounted max over actions of value of next state
			double maxActionValue = -1000000;
			for(unsigned int a = 0; a < numActions; a++)
			{
				//calculate expected utility of taking action a in state s1
				unsigned int s2 =  getNextState(s1,a);
				double expUtil = T[s1][a][s2] * V[s2];
				if(expUtil > maxActionValue)
					maxActionValue = expUtil;
			}
			new_V[s1] += discount * maxActionValue;

			//update delta to track convergence
			double absDiff = abs(new_V[s1] - V[s1]);
			if(absDiff > delta) delta = absDiff;
		}

		for(unsigned int s = 0; s < numStates; s++) V[s] = new_V[s];

	}
	while( delta > eps * (1 - discount) / discount);
	delete new_V;
}


void FeatureGridMDP::valueIteration(double eps, double* input_V)
{
	Vinitialized = true;
	double delta = 0;
	for(unsigned int s = 0; s < numStates; s++)
	{
		V[s] = input_V[s];
	}
	double* new_V = new double[numStates];
	for(unsigned int s = 0; s < numStates; s++) new_V[s] = R[s];
	//repeat until convergence within error eps
	do
	{
		delta = 0;
		for(unsigned int s1 = 0; s1 < numStates; s1++)
		{
			new_V[s1] = R[s1];
			//add discounted max over actions of value of next state
			double maxActionValue = -1000000;
			for(unsigned int a = 0; a < numActions; a++)
			{
				//calculate expected utility of taking action a in state s1
				unsigned int s2 =  getNextState(s1,a);
				double expUtil = T[s1][a][s2] * V[s2];
				if(expUtil > maxActionValue)
					maxActionValue = expUtil;
			}
			new_V[s1] += discount * maxActionValue;

			//update delta to track convergence
			double absDiff = abs(new_V[s1] - V[s1]);
			if(absDiff > delta) delta = absDiff;
		}

		for(unsigned int s = 0; s < numStates; s++) V[s] = new_V[s];

	}
	while( delta > eps * (1 - discount) / discount);
	delete new_V;
}

void FeatureGridMDP::displayFeatureWeights()
{
	ios::fmtflags f( cout.flags() );
	std::streamsize ss = std::cout.precision();
	cout << setiosflags(ios::fixed) << setprecision(4);
	for(int i=0; i<numFeatures; i++)
		cout << featureWeights[i] << " ";
	cout << endl;
	cout.flags( f );
	cout << setprecision(ss);
}

bool GridMDP::isOptimalAction(unsigned int state, unsigned int action, double tolerance)
{
	assert(Qinitialized);
	double max_q = lowest_double;
	for(unsigned int a = 0; a < numActions; a++)
	{
		if( Q[state][a] > max_q )
		{
			max_q  = Q[state][a];
		}
	}
	if( abs(Q[state][action] - max_q) < tolerance) return true;
	return false;

}

void GridMDP::getOptimalPolicy(vector<unsigned int> & opt)
{
	if(!Qinitialized)
		calculateQValues();
	for(unsigned int s = 0; s < numStates; s++)
	{
		double max_q = lowest_double;
		unsigned int max_action = 0;
		vector<unsigned int> vactions = getValidActions(s);
		for(unsigned int a : vactions)
		{
			if( Q[s][a] > max_q )
			{
				max_q  = Q[s][a];
				max_action = a;
			}
		}
		opt[s] = max_action;      
	}
}

vector<vector<double> > GridMDP::getOptimalStochasticPolicy()
{
	vector<vector<double> > opt_stochastic;

	if(!Qinitialized)
		calculateQValues();
	for(unsigned int s = 0; s < numStates; s++)
	{
		vector<double> action_probs(numActions);
		opt_stochastic.push_back(action_probs);
		double cum_sum = 0.0;
		for(unsigned int a = 0; a < numActions; a++)
		{
			if(isOptimalAction(s,a))
			{
				opt_stochastic[s][a] = 1.0;
				cum_sum += 1.0;
			}
			else
				opt_stochastic[s][a] = 0.0;
		}
		//normalize
		for(unsigned int a = 0; a < numActions; a++)
			opt_stochastic[s][a] = opt_stochastic[s][a] / cum_sum;    
	}
	return opt_stochastic;
}


void MDP::calculateQValues()
{
	assert(R != nullptr); 
	assert(T != nullptr);

	Qinitialized = true;
	for(unsigned int s = 0; s < numStates; s++)
	{
		for(unsigned int a = 0; a < numActions; a++)
		{
			Q[s][a] = getReward(s);
			for(unsigned int s2 = 0; s2 < numStates; s2++)
			{
				Q[s][a] += discount * T[s][a][s2] * V[s2];
			}
		}
	}


}


void GridMDP::displayPolicy(vector<unsigned int> & policy)
{
	unsigned int count = 0;
	for(unsigned int r = 0; r < gridHeight; r++)
	{
		for(unsigned int c = 0; c < gridWidth; c++)
		{
			if(isTerminalState(count)) cout << "*" << "  ";
			else if(isWallState(count)) cout << "w" << "  ";
			else if(policy[count]==0) cout << "^" << "  ";
			else if(policy[count]==1) cout << "v" << "  ";
			else if(policy[count]==2) cout << "<" << "  ";
			else cout << ">" << "  ";
			count++;
		}
		cout << endl;
	}
}



void GridMDP::displayTransitions()
{
	ios::fmtflags f( cout.flags() );
	std::streamsize ss = std::cout.precision();

	cout << "-------- UP ----------" << endl;
	for(unsigned int s1 = 0; s1 < numStates; s1++)
	{
		for(unsigned int s2 = 0; s2 < numStates; s2++)
		{
			cout << setiosflags(ios::fixed)
				<< setprecision(2)
				<< T[s1][UP][s2] << " ";
		}
		cout << endl;
	}
	cout << "-------- DOWN ----------" << endl;
	for(unsigned int s1 = 0; s1 < numStates; s1++)
	{
		for(unsigned int s2 = 0; s2 < numStates; s2++)
		{
			cout << T[s1][DOWN][s2] << " ";
		}
		cout << endl;
	}
	cout << "-------- LEFT ----------" << endl;
	for(unsigned int s1 = 0; s1 < numStates; s1++)
	{
		for(unsigned int s2 = 0; s2 < numStates; s2++)
		{
			cout << T[s1][LEFT][s2] << " ";
		}
		cout << endl;
	}
	cout << "-------- RIGHT ----------" << endl;
	for(unsigned int s1 = 0; s1 < numStates; s1++)
	{
		for(unsigned int s2 = 0; s2 < numStates; s2++)
		{
			cout << T[s1][RIGHT][s2] << " ";
		}
		cout << endl;
	}
	cout.flags( f );
	cout << setprecision(ss);
}

void GridMDP::displayQValues()
{
	ios::fmtflags f( cout.flags() );
	std::streamsize ss = std::cout.precision();
	if (Q == nullptr){
		cout << "ERROR: no Q values!" << endl;
		return;
	}
	// assert(Qinitialized);
	cout << "-------- UP ----------" << endl;
	for(unsigned int r = 0; r < gridHeight; r++)
	{
		for(unsigned int c = 0; c < gridWidth; c++)
		{
			unsigned int state = r*gridWidth + c;
			cout << setiosflags(ios::fixed)
				<< setprecision(3)
				<< Q[state][UP] << "  ";
		}
		cout << endl;
	}
	cout << "-------- DOWN ----------" << endl;
	for(unsigned int r = 0; r < gridHeight; r++)
	{
		for(unsigned int c = 0; c < gridWidth; c++)
		{
			unsigned int state = r*gridWidth + c;
			cout << Q[state][DOWN] << "  ";
		}
		cout << endl;
	}

	cout << "-------- LEFT ----------" << endl;
	for(unsigned int r = 0; r < gridHeight; r++)
	{
		for(unsigned int c = 0; c < gridWidth; c++)
		{
			unsigned int state = r*gridWidth + c;
			cout << Q[state][LEFT] << "  ";
		}
		cout << endl;
	}

	cout << "-------- RIGHT ----------" << endl;
	for(unsigned int r = 0; r < gridHeight; r++)
	{
		for(unsigned int c = 0; c < gridWidth; c++)
		{
			unsigned int state = r*gridWidth + c;
			cout << Q[state][RIGHT] << "  ";
		}
		cout << endl;
	}
	cout.flags( f );
	cout << setprecision(ss);
}


void MDP::initRandDeterministicPolicy(vector<unsigned int> & policy)
{
	for(unsigned int s = 0; s < numStates; s++)
	{
		policy[s] = rand() % numActions;
	}
}


void MDP::deterministicPolicyIteration(vector<unsigned int> & policy)
{
	//generate random policy
	initRandDeterministicPolicy(policy);
	bool policyUnchanged = false;
	while(!policyUnchanged)
	{
		//update values based on current policy
		deterministicPolicyEvaluation(policy); //uses default value of k=20 for now
		//run policy improvement
		policyUnchanged = true;
		for(unsigned int s1 = 0; s1 < numStates; s1++)
		{
			//find expected utility of best action in s1 (undiscounted and reward in s1 doesn't matter ) (see R&N 657)
			//along the way also keep track of expected utility of taking policy action
			//also remember the best action
			double maxActionValue = -10000;
			double policyActionValue = 0;
			unsigned int bestAction = -1;
			for(unsigned int a = 0; a < numActions; a++)
			{
				//calculate expected utility of taking action a in state s1
				double expUtil = 0;
				for(unsigned int s2 = 0; s2 < numStates; s2++)
				{
					expUtil += T[s1][a][s2] * V[s2];
				}
				if(expUtil > maxActionValue)
				{
					bestAction = a;
					maxActionValue = expUtil;
				}
				//remember how well current policy action does
				if(a == policy[s1])
					policyActionValue = expUtil;
			}
			//check if policy needs to be updated
			if(maxActionValue > policyActionValue)
			{
				policy[s1] = bestAction;
				policyUnchanged = false;
			}
		}
	}
}

//currently just starts with CTR policy
void MDP::deterministicPolicyIteration(vector<unsigned int> & policy, int steps)
{
	//generate random policy
	//generate CTR policy
	for(unsigned int s = 0; s < numStates; s++)
	{
		policy[s] = 3;
	}
	bool policyUnchanged = false;
	for(int step = 0; step < steps; step++)
	{
		//update values based on current policy
		deterministicPolicyEvaluation(policy); //uses default value of k=20 for now
		//run policy improvement
		policyUnchanged = true;
		for(unsigned int s1 = 0; s1 < numStates; s1++)
		{
			//find expected utility of best action in s1 (undiscounted and reward in s1 doesn't matter ) (see R&N 657)
			//along the way also keep track of expected utility of taking policy action
			//also remember the best action
			double maxActionValue = -10000;
			double policyActionValue = 0;
			unsigned int bestAction = -1;
			for(unsigned int a = 0; a < numActions; a++)
			{
				//calculate expected utility of taking action a in state s1
				double expUtil = 0;
				for(unsigned int s2 = 0; s2 < numStates; s2++)
				{
					expUtil += T[s1][a][s2] * V[s2];
				}
				if(expUtil > maxActionValue)
				{
					bestAction = a;
					maxActionValue = expUtil;
				}
				//remember how well current policy action does
				if(a == policy[s1])
					policyActionValue = expUtil;
			}
			//check if policy needs to be updated
			if(maxActionValue > policyActionValue)
			{
				policy[s1] = bestAction;
				policyUnchanged = false;
			}
		}
		if(policyUnchanged)
		{
			cout << "unchanged" << endl;
			break;
		}
	}
}








//see how well a specific policy performs on mdp by updating state values k times
void MDP::deterministicPolicyEvaluation(vector<unsigned int> & policy, int k)
{
	for(int iter = 0; iter < k; iter++)
	{
		//update value of each state
		for(unsigned int s1 = 0; s1 < numStates; s1++)
		{
			double tempV = 0;
			//add reward
			tempV += getReward(s1);
			//add discounted expected value after taking policy action
			unsigned int policyAction = policy[s1];
			double expUtil = 0;
			for(unsigned int s2 = 0; s2 < numStates; s2++)
			{
				expUtil += T[s1][policyAction][s2] * V[s2];
			}
			tempV += discount * expUtil;
			V[s1] = tempV;
		}
	}
}

//runs value iteration, Note: it sets V to zero at start so won't work with a warm start
void MDP::valueIteration(double eps)
{
	Vinitialized = true;
	//initialize values to zero
	double delta;

	//repeat until convergence within error eps
	do
	{
		delta = 0;
		//update value of each state
		for(unsigned int s1 = 0; s1 < numStates; s1++)
		{
			double tempV = 0;
			//add reward
			tempV += getReward(s1);
			//add discounted max over actions of value of next state
			double maxActionValue = -10000;
			for(unsigned int a = 0; a < numActions; a++)
			{
				//calculate expected utility of taking action a in state s1
				double expUtil = 0;
				for(unsigned int s2 = 0; s2 < numStates; s2++)
				{
					expUtil += T[s1][a][s2] * V[s2];
				}
				if(expUtil > maxActionValue)
					maxActionValue = expUtil;
			}
			tempV += discount * maxActionValue;

			//update delta to track convergence
			double absDiff = abs(tempV - V[s1]);
			if(absDiff > delta)
				delta = absDiff;
			V[s1] = tempV;
		}

	}
	while(delta > eps );
}

void MDP::valueIteration(double eps, double* input_v) //warm start
{
	Vinitialized = true;
	//initialize values to zero
	double delta;

	for(unsigned int s = 0; s < numStates; s++)
	{
		V[s] = input_v[s];
	}
	//repeat until convergence within error eps
	do
	{
		delta = 0;
		//update value of each state
		for(unsigned int s1 = 0; s1 < numStates; s1++)
		{
			double tempV = 0;
			//add reward
			tempV += getReward(s1);
			//add discounted max over actions of value of next state
			double maxActionValue = -10000;
			for(unsigned int a = 0; a < numActions; a++)
			{
				//calculate expected utility of taking action a in state s1
				double expUtil = 0;
				for(unsigned int s2 = 0; s2 < numStates; s2++)
				{
					expUtil += T[s1][a][s2] * V[s2];
				}
				if(expUtil > maxActionValue)
					maxActionValue = expUtil;
			}
			tempV += discount * maxActionValue;

			//update delta to track convergence
			double absDiff = abs(tempV - V[s1]);
			if(absDiff > delta)
				delta = absDiff;
			V[s1] = tempV;
		}

	}
	while(delta > eps );
}

void GridMDP::setDeterministicGridTransitions() //specific to grid MDP
{

	//Transition matrices for actions
	//UP
	for(unsigned int s = 0; s < numStates; s++)
	{
		if((s >= gridWidth) && !isWallState(s - gridWidth))
			T[s][UP][s - gridWidth] = 1.0;

		else
			T[s][UP][s] = 1.0;

	}
	//DOWN
	for(unsigned int s = 0; s < numStates; s++)
	{
		if((s < (gridHeight - 1) * gridWidth) && !isWallState(s + gridWidth))
			T[s][DOWN][s + gridWidth] = 1.0;
		else
			T[s][DOWN][s] = 1.0;

	}
	//LEFT
	for(unsigned int s = 0; s < numStates; s++)
	{
		if(((s % gridWidth) > 0) && !isWallState(s-1))
			T[s][LEFT][s - 1] = 1.0;
		else
			T[s][LEFT][s] = 1.0;

	}
	//RIGHT
	for(unsigned int s = 0; s < numStates; s++)
	{
		if((s % gridWidth < gridWidth - 1) && !isWallState(s+1))
			T[s][RIGHT][s + 1] = 1.0;
		else
			T[s][RIGHT][s] = 1.0;

	}

	//Terminals
	if(terminalStates != nullptr) 
	{
		//cout << "setting up terminals" << endl;
		for(unsigned int s = 0; s < numStates; s++)
		{
			if(terminalStates[s])
			{
				//cout << "for state " << s << endl;
				if(s >= gridWidth) T[s][UP][s - gridWidth] = 0.0;
				if(s < (gridHeight - 1) * gridWidth) T[s][DOWN][s + gridWidth] = 0.0;
				if(s % gridWidth > 0) T[s][LEFT][s - 1] = 0.0;
				if(s % gridWidth < gridWidth - 1) T[s][RIGHT][s + 1] = 0.0;
				T[s][UP][s] = 0.0;
				T[s][DOWN][s] = 0.0;
				T[s][LEFT][s] = 0.0; 
				T[s][RIGHT][s] = 0.0;
			}
		}
	}

	//set wall states to have zero transitions in or out
	if(wallStates != nullptr) 
	{
		//cout << "setting up terminals" << endl;
		for(unsigned int s = 0; s < numStates; s++)
		{
			if(wallStates[s])
			{
				//no transitions outgoing
				for(unsigned int s2 = 0; s2 < numStates; s2++)
				{
					T[s][UP][s2] = 0.0;
					T[s][DOWN][s2] = 0.0;
					T[s][LEFT][s2] = 0.0; 
					T[s][RIGHT][s2] = 0.0;

				}
			}
		}
	}


	//check that all state transitions add up properly!
	for(unsigned int s = 0; s < numStates; s++)
	{
		//cout << "state " << s << endl;
		for(unsigned int a = 0; a < numActions; a++)
		{
			//cout << "action " << a << endl;
			//add up transitions
			double sum = 0;
			for(unsigned int s2 = 0; s2 < numStates; s2++)

				sum += T[s][a][s2];
			//cout << sum << endl;
			assert(sum <= 1.0);
		}
	}

}

void GridMDP::setStochasticGridTransitions() //specific to grid MDP
{

	//Transition matrices for actions
	//UP
	for(unsigned int s = 0; s < numStates; s++)
	{

		//go forward
		if(s >= gridWidth)
			T[s][UP][s - gridWidth] = 0.7;
		else    
			T[s][UP][s] = 0.7;
		//posibility of going left
		if(s % gridWidth == 0) 
			T[s][UP][s] = 0.15;
		else
			T[s][UP][s-1] = 0.15;
		//possibility of going right
		if(s % gridWidth < gridWidth - 1)
			T[s][UP][s+1] = 0.15;
		else
			T[s][UP][s] = 0.15;

		//check top left corner case
		if(s < gridWidth && s % gridWidth == 0)
			T[s][UP][s] = 0.7 + 0.15;
		//check top right corner case
		else if((s < gridWidth) && (s % gridWidth == gridWidth - 1))
			T[s][UP][s] = 0.7 + 0.15;
	}
	//DOWN
	for(unsigned int s = 0; s < numStates; s++)
	{
		//go down
		if(s < (gridHeight - 1) * gridWidth)
			T[s][DOWN][s + gridWidth] = 0.7;
		else
			T[s][DOWN][s] = 0.7;
		//posibility of going left
		if(s % gridWidth == 0) 
			T[s][DOWN][s] = 0.15;
		else
			T[s][DOWN][s-1] = 0.15;
		//possibility of going right
		if(s % gridWidth < gridWidth - 1)
			T[s][DOWN][s+1] = 0.15;
		else
			T[s][DOWN][s] = 0.15;

		//check bottom left corner case
		if(s >= (gridHeight - 1) * gridWidth && s % gridWidth == 0)
			T[s][DOWN][s] = 0.7 + 0.15;
		//check bottom right corner case
		else if(s >= (gridHeight - 1) * gridWidth && s % gridWidth == gridWidth - 1)
			T[s][DOWN][s] = 0.7 + 0.15;
	}   
	//LEFT
	for(unsigned int s = 0; s < numStates; s++)
	{
		//go left
		if(s % gridWidth > 0)
			T[s][LEFT][s - 1] = 0.7;
		else
			T[s][LEFT][s] = 0.7;
		//go up
		if(s >= gridWidth)
			T[s][LEFT][s - gridWidth] = 0.15;
		else
			T[s][LEFT][s] = 0.15;
		//go down
		if(s < (gridHeight - 1) * gridWidth)
			T[s][LEFT][s + gridWidth] = 0.15;
		else
			T[s][LEFT][s] = 0.15;

		//check top left corner case
		if(s < gridWidth && s % gridWidth == 0)
			T[s][LEFT][s] = 0.7 + 0.15;
		//check bottom left corner case
		else if(s >= (gridHeight - 1) * gridWidth && s % gridWidth == 0)
			T[s][LEFT][s] = 0.7 + 0.15;


	}
	//RIGHT
	for(unsigned int s = 0; s < numStates; s++)
	{
		if(s % gridWidth < gridWidth - 1)
			T[s][RIGHT][s + 1] = 0.7;
		else
			T[s][RIGHT][s] = 0.7;
		//go up
		if(s >= gridWidth)
			T[s][RIGHT][s - gridWidth] = 0.15;
		else
			T[s][RIGHT][s] = 0.15;
		//go down
		if(s < (gridHeight - 1) * gridWidth)
			T[s][RIGHT][s + gridWidth] = 0.15;
		else
			T[s][RIGHT][s] = 0.15;
		//check top right corner case
		if((s < gridWidth) && (s % gridWidth == gridWidth - 1))
			T[s][RIGHT][s] = 0.7 + 0.15;
		//check bottom right corner case
		else if(s >= (gridHeight - 1) * gridWidth && s % gridWidth == gridWidth - 1)
			T[s][RIGHT][s] = 0.7 + 0.15;

	}

	//Terminals
	if(terminalStates != nullptr) 
	{
		//cout << "setting up terminals" << endl;
		for(unsigned int s = 0; s < numStates; s++)
		{
			if(terminalStates[s])
			{
				for(unsigned int s1 = 0; s1 < numStates; s1++)
				{
					//set all outgoing transitions to zero
					T[s][UP][s1] = 0.0;
					T[s][DOWN][s1] = 0.0;
					T[s][LEFT][s1] = 0.0;
					T[s][RIGHT][s1] = 0.0;
				}
			}
		}
	}

	//set wall states to have zero transitions in or out
	if(wallStates != nullptr) 
	{
		//cout << "setting up terminals" << endl;
		for(unsigned int s = 0; s < numStates; s++)
		{
			if(wallStates[s])
			{
				//no transitions outgoing
				for(unsigned int s2 = 0; s2 < numStates; s2++)
				{
					T[s][UP][s2] = 0.0;
					T[s][DOWN][s2] = 0.0;
					T[s][LEFT][s2] = 0.0; 
					T[s][RIGHT][s2] = 0.0;

				}
			}
		}
	}

	//check that no states have more than 1.0 prob of transition!
	for(unsigned int s = 0; s < numStates; s++)
	{
		cout << "state " << s << endl;
		for(unsigned int a = 0; a < numActions; a++)
		{
			cout << "action " << a << endl;
			//add up transitions
			double sum = 0;
			for(unsigned int s2 = 0; s2 < numStates; s2++)

				sum += T[s][a][s2];
			cout << sum << endl;
			assert(sum <= 1.0);
		}
	}

}


void GridMDP::displayRewards()
{
	ios::fmtflags f( cout.flags() );
	streamsize ss = std::cout.precision();
	assert(R != nullptr);
	cout << setiosflags(ios::fixed) << setprecision(2);

	for(unsigned int r = 0; r < gridHeight; r++)
	{
		for(unsigned int c = 0; c < gridWidth; c++)
		{
			unsigned int state = r*gridWidth + c;
			cout << getReward(state) << "  ";
		}
		cout << endl;
	}
	cout.flags(f);
	cout << setprecision(ss);
}

void GridMDP::displayValues()
{
	ios::fmtflags f( cout.flags() );
	std::streamsize ss = std::cout.precision();

	assert(R != nullptr);
	for(unsigned int r = 0; r < gridHeight; r++)
	{
		for(unsigned int c = 0; c < gridWidth; c++)
		{
			unsigned int state = r*gridWidth + c;
			cout << setiosflags(ios::fixed)
				<< setprecision(3)
				<< V[state] << "  ";
		}
		cout << endl;
	}
	cout.flags(f);
	cout << setprecision(ss);
}



bool operator== (MDP & lhs, MDP & rhs)
{
	double* R1 = lhs.getRewards();
	double* R2 = rhs.getRewards();
	for(unsigned int s = 0; s < lhs.getNumStates(); s++){
		if(R1[s] != R2[s]) return false;
	}
	return true;

}


//computes the dot product between two vectors
double dotProduct(double x[], double y[], int length)
{
	double result = 0;
	for(int i=0; i<length; i++)
		result += x[i] * y[i];
	return result;
}

//    """compute a rollout starting in start_state s0 of length L
//at every state an action is chosen as argmax on Q-values (random tie-breaks)
//       and a transition is sampled from. This continues for L iterations
//       or until a terminal is reached"""
// uses Q-values 
vector<pair<unsigned int, unsigned int>> MDP::monte_carlo_argmax_rollout(int s0, int L)
{
	//   assert(MDP::isQInitialized());
	//   assert(MDP::isVInitialized());
	vector<pair<unsigned int,unsigned int>> rollout;

	int state = s0;
	int count = 0;
	while(!MDP::isTerminalState(state) && count < L)
	{
		//sample from actions based on q_values
		vector<double> state_q_vals(NUM_ACTIONS);
		for(int a = 0; a < NUM_ACTIONS; a++)
			state_q_vals[a] = getQValue(state, a);
		vector<int> best_actions =  argmax_all(state_q_vals);
		int a = best_actions[rand() % best_actions.size()];
		rollout.push_back(make_pair(state, a));
		double*** T = getTransitions();
		vector<double> trans_probs;
		vector<int> reachable_states;
		for(unsigned int s = 0; s < getNumStates(); s++)
		{
			if(T[state][a][s] > 0)
			{
				trans_probs.push_back(T[state][a][s]);
				reachable_states.push_back(s);
			}
		}
		int next_state_idx = roulette_wheel(trans_probs);
		state = reachable_states[next_state_idx];
		count += 1;
	}
	//if rollout ends in terminal add terminal and 0 action
	if(MDP::isTerminalState(state))
		rollout.push_back(make_pair(state,0));

	return rollout;
}


//    """compute a rollout starting in start_state s0 of length L
// 1-epsilon percent of the time at every state an action is chosen as argmax on Q-values (random tie-breaks), and epsilon percent of the time a random action is taken
//       and a transition is sampled from. This continues for L iterations
//       or until a terminal is reached"""
// uses Q-values 
vector<pair<unsigned int, unsigned int>> MDP::epsilon_random_rollout(int s0, int L, double epsilon)
{
	assert(MDP::isQInitialized());
	assert(MDP::isVInitialized());
	vector<pair<unsigned int,unsigned int>> rollout;

	int state = s0;
	int count = 0;
	while(!MDP::isTerminalState(state) && count < L)
	{
		double r = ((double) rand() / (RAND_MAX));
		int a;
		if(r > epsilon)
		{
			//sample from actions based on q_values
			vector<double> state_q_vals(NUM_ACTIONS);
			for(int a = 0; a < NUM_ACTIONS; a++)
				state_q_vals[a] = getQValue(state, a);
			vector<int> best_actions =  argmax_all(state_q_vals);
			a = best_actions[rand() % best_actions.size()];
		}
		else{
			//sample action randomly
			a = rand() % NUM_ACTIONS;

		}
		rollout.push_back(make_pair(state, a));
		double*** T = getTransitions();
		vector<double> trans_probs;
		vector<int> reachable_states;
		for(unsigned int s = 0; s < getNumStates(); s++)
		{
			if(T[state][a][s] > 0)
			{
				trans_probs.push_back(T[state][a][s]);
				reachable_states.push_back(s);
			}
		}
		int next_state_idx = roulette_wheel(trans_probs);
		state = reachable_states[next_state_idx];
		count += 1;
	}
	//if rollout ends in terminal add terminal and 0 action
	if(MDP::isTerminalState(state))
		rollout.push_back(make_pair(state,0));

	return rollout;
}

//"""given a list of positive weights w, return a random index weighted by w"""  
int roulette_wheel(vector<double> w)
{
	int num_weights = w.size();
	for(double w_ : w)
		assert(w_ >= 0);
	//calculate sum
	double w_sum = 0;
	for(double w_ : w)
		w_sum += w_;
	//calculate cumulative sum
	double w_cumsum[num_weights];
	for(int i=0; i<num_weights;i++) w_cumsum[i] = 0;

	for(int i=0; i<num_weights;i++)
		for(int j=i; j<num_weights;j++)
			w_cumsum[j] += w[i];

	double rand_value = rand() / (double) RAND_MAX;
	double roulette_spin = rand_value * w_sum;
	int indx = 0;
	while(w_cumsum[indx] < roulette_spin)
		indx += 1;
	return indx;
}    

vector<int> argmax_all(vector<double> xs)
{
	vector<int> arg_maxs;
	//find max value 
	double max_val = lowest_double;
	for(double x : xs)
	{
		if(x > max_val)
			max_val = x;
	}
	for(unsigned int i = 0; i < xs.size(); i++)
		if(isEqual(xs[i], max_val))
			arg_maxs.push_back(i);
	return arg_maxs;
}




#endif
