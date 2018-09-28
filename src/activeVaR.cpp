#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>

#include "../include/mdp.hpp"
#include "../include/grid_domain.hpp"
#include "../include/confidence_bounds.hpp"
#include "../include/unit_norm_sampling.hpp"
#include "../include/feature_birl.hpp"
#include "../include/feature_gain.hpp"

#define SIZE 8  //Gridworld width and height
#define FEATURES 48 //number of features 

#define INTERACTIONS 30 // number of active learning sessions

#define QUERY_TYPE 1 // 0-> action query, 1 -> critique query
#define PATH_LENGTH 1     // active query path length for critique queries

#define CHAIN_LENGH 10000 // MCMC chain length
#define STEP_SIZE 0.05  // reward sampling stepsize

#define ALPHA 100  // confidence factor  -> 100 since we use groundtruth labeler
#define DISCOUNT 0.95 // discount factor for RL


using namespace std;


template <typename T1, typename T2>
struct less_second {
	typedef pair<T1, T2> type;
	bool operator ()(type const& a, type const& b) const {
		return a.second < b.second;
	}
};

FeatureGridMDP* makeWorld()
{
	FeatureGridMDP* fmdp = nullptr;

	const int numFeatures = FEATURES; 
	const int numStates = SIZE * SIZE;
	const int width = SIZE ;
	const int height = SIZE ;
	double gamma = DISCOUNT;
	vector<unsigned int> initStates;
	for(int i=0;i<numStates;i++)
		initStates.push_back(i); // can start anywhere
	vector<unsigned int> termStates = {};
	bool stochastic = false;

	//create random world 
	double** stateFeatures = initRandomFeaturesRandomDomain(numStates, numFeatures);
	double featureWeights[numFeatures];

	double total_w = 0.0;
	for(unsigned int fi=0; fi<numFeatures; fi++)
	{
		featureWeights[fi] = pow(-1,rand())*(rand()%100)/10.0;
		total_w += abs(featureWeights[fi]);
	}
	cout << "-- Groundtruth feature weights -- \n";
	for(unsigned int fi=0; fi < numFeatures; fi++)
	{
		featureWeights[fi] /= total_w; 
		cout << featureWeights[fi] << ", ";
	}
	cout << endl;
	fmdp = new FeatureGridMDP(width, height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
	return fmdp;

}

double policyLoss(vector<unsigned int> policy,  FeatureGridMDP * mdp, bool count=false)
{
	if (count) // binary overlap used in AS
	{
		unsigned int count = 0;
		mdp -> calculateQValues();
		for(unsigned int i=0; i < policy.size(); i++)
		{
			if(! mdp->isOptimalAction(i,policy[i])) {
				count++;
			}
		} 
		return (double)count/(double)policy.size()*100; 
	}
	else // expected return
	{
		double diff =  getExpectedReturn(mdp) - evaluateExpectedReturn(policy, mdp, 0.001);
		return diff;

	}
}

int main(int argc, char** argv) 
{
	if (argc < 2) {
		cout << "[WARN] No random seed provided!" << endl;
		srand(rand());
	}
	else
		srand (time(NULL)+atoi(argv[1]));

	double losses[INTERACTIONS+1]  = {0};  // activeVaR
	double losses1[INTERACTIONS+1]  = {0}; // active sampling
	double losses2[INTERACTIONS+1]  = {0}; // active reward learning from critiques  
	double losses3[INTERACTIONS+1]  = {0}; // uniform sampling / random

	double VaR = 0.95;

	double alpha = ALPHA;                     //confidence param for BIRL
	const unsigned int chain_length = CHAIN_LENGH;   //length of MCMC chain
	double step = STEP_SIZE; 
	const double min_r = -1;
	const double max_r = 1; 
	double eps = 0.001;

	FeatureGridMDP* fmdp = makeWorld();
	FeatureGridMDP* fmdp1 = fmdp->deepcopy();
	FeatureGridMDP* fmdp2 = fmdp->deepcopy();
	FeatureGridMDP* fmdp3 = fmdp->deepcopy();

	vector<unsigned int> initStates;
	for(unsigned int s = 0; s < fmdp->getNumStates(); s++)
		if(fmdp->isInitialState(s))
			initStates.push_back(s);

	///  solve mdp for weights and get optimal policyLoss
	vector<unsigned int> opt_policy (fmdp->getNumStates());
	fmdp->valueIteration(eps);
	fmdp1->valueIteration(eps);
	fmdp2->valueIteration(eps);
	fmdp3->valueIteration(eps);


	cout << "-- rewards --" << endl;
	fmdp->displayRewards();

	fmdp->calculateQValues();
	fmdp1->calculateQValues();
	fmdp2->calculateQValues();
	fmdp3->calculateQValues();

	fmdp->getOptimalPolicy(opt_policy);
	cout << "-- optimal policy --" << endl;
	fmdp->displayPolicy(opt_policy);

	///  generate initial demo -> set to 2 random state-actions
	vector<pair<unsigned int,unsigned int> > good_demos;
	unsigned int s = rand() % fmdp->getNumStates();
	for( unsigned int i = 0; i < 2; i++)
	{
		good_demos.push_back(make_pair(s,opt_policy[s]));
		unsigned int ns = fmdp->getNextState(s,opt_policy[s]);  
		if(ns == s) s = rand()% fmdp->getNumStates();
		else s = ns;
	}	

	///  run BIRL to get chain and Map policyLoss 
	//give it a copy of mdp to initialize
	FeatureBIRL birl(fmdp, min_r, max_r, chain_length, step, alpha); 
	birl.addPositiveDemos(good_demos);
	FeatureBIRL birl1(fmdp1, min_r, max_r, chain_length, step, alpha); 
	birl1.addPositiveDemos(good_demos);
	FeatureBIRL birl2(fmdp2, min_r, max_r, chain_length, step, alpha); 
	birl2.addPositiveDemos(good_demos);
	FeatureBIRL birl3(fmdp3, min_r, max_r, chain_length, step, alpha); 
	birl3.addPositiveDemos(good_demos);

	FeatureInfoGainCalculator gain_computer = FeatureInfoGainCalculator(&birl1);
	FeatureInfoGainCalculator gain_computer2 = FeatureInfoGainCalculator(&birl2);

	for(unsigned int itr = 0; itr < INTERACTIONS; itr++)
	{

		birl.displayDemos();
		cout << "Running BIRL for ActiveVaR.... " << endl;
		birl.run();

		birl1.displayDemos();
		cout << "Running BIRL for AS.... " << endl;
		birl1.run();

		birl2.displayDemos();
		cout << "Running BIRL for ARC.... " << endl;
		birl2.run();

		birl3.displayDemos();
		cout << "Running BIRL for Random.... " << endl;
		birl3.run();

		FeatureGridMDP* mapMDP = birl.getMAPmdp();
		cout << "ActiveVaR Recovered weights:" << endl;
		mapMDP->displayFeatureWeights();
		cout << "ActiveVaR Recovered reward:" << endl;
		mapMDP->displayRewards();

		vector<unsigned int> map_policy  (mapMDP->getNumStates());
		mapMDP->valueIteration(0.001);
		mapMDP->deterministicPolicyIteration(map_policy);
		mapMDP->displayPolicy(map_policy);
		double loss = policyLoss(map_policy, fmdp);
		cout.precision(4);

		FeatureGridMDP* mapMDP1 = birl1.getMAPmdp();
		cout << "AS Recovered weights:" << endl;
		mapMDP1->displayFeatureWeights();
		cout << "AS Recovered reward:" << endl;
		mapMDP1->displayRewards();

		vector<unsigned int> map_policy1  (mapMDP1->getNumStates());
		mapMDP1->valueIteration(0.001);
		mapMDP1->deterministicPolicyIteration(map_policy1);
		mapMDP1->displayPolicy(map_policy1);
		double loss1 = policyLoss(map_policy1, fmdp1);
		cout.precision(4);

		FeatureGridMDP* mapMDP2 = birl2.getMAPmdp();
		cout << "ARC Recovered weights:" << endl;
		mapMDP2->displayFeatureWeights();
		cout << "ARC Recovered reward:" << endl;
		mapMDP2->displayRewards();

		vector<unsigned int> map_policy2  (mapMDP2->getNumStates());
		mapMDP2->valueIteration(0.001);
		mapMDP2->deterministicPolicyIteration(map_policy2);
		mapMDP2->displayPolicy(map_policy2);
		double loss2 = policyLoss(map_policy2, fmdp2);
		cout.precision(4);

		FeatureGridMDP* mapMDP3 = birl3.getMAPmdp();
		cout << "Random Recovered weights:" << endl;
		mapMDP3->displayFeatureWeights();
		cout << "Random Recovered reward:" << endl;
		mapMDP3->displayRewards();

		vector<unsigned int> map_policy3  (mapMDP3->getNumStates());
		mapMDP3->valueIteration(0.001);
		mapMDP3->deterministicPolicyIteration(map_policy3);
		mapMDP3->displayPolicy(map_policy3);
		double loss3 = policyLoss(map_policy3, fmdp3);
		cout.precision(4);

		cout << "Current activeVaR policy loss: "  << loss << "%" << endl;
		losses[itr] = loss;
		cout << "Current AS policy loss1: "  << loss1 << "%" << endl;
		losses1[itr] = loss1;
		cout << "Current ARC policy loss2: "  << loss2 << "%" << endl;
		losses2[itr] = loss2;
		cout << "Current Random policy loss3: "  << loss3 << "%" << endl;
		losses3[itr] = loss3;

		unsigned int numStates = fmdp->getNumStates();

		// --- Active VaR -----
		cout << "------------"<< endl;
		//Get V^* and V^\pi_eval for each start state 
		vector<vector<double>> evds(initStates.size(), vector<double>(chain_length));
		for(unsigned int i=0; i<chain_length; i++)
		{
			GridMDP* sampleMDP = (*(birl.getRewardChain() + i));
			vector<unsigned int> sample_pi(sampleMDP->getNumStates());
			sampleMDP->getOptimalPolicy(sample_pi);
			vector<double> Vstar_vec =  getExpectedReturnVector(sampleMDP);
			vector<double> Vhat_vec = evaluateExpectedReturnVector(map_policy, sampleMDP, eps);
			for(unsigned int j = 0; j < Vstar_vec.size(); j++)
			{
				double EVDiff = Vstar_vec[j] - Vhat_vec[j];
				evds[j][i] = EVDiff;
			}
		}    
		clock_t c_start = clock();
		cout << "VaR:" << endl;
		cout.precision(4);
		unsigned int query_state = rand()%numStates;
		double max_VaR = 0;
		pair<unsigned int, unsigned int> best_pair; 
		for(unsigned int s = 0; s < evds.size(); s++)
		{
			std::sort(evds[s].begin(), evds[s].end());
			int VaR_index = (int) chain_length * VaR;
			double eval_VaR = evds[s][VaR_index];  
			if (eval_VaR > max_VaR)
			{
				if(QUERY_TYPE) best_pair = make_pair(s, map_policy[s]);
				else best_pair = make_pair(s, opt_policy[s]);
			}   
			if(s % fmdp->getGridWidth() < fmdp->getGridWidth() - 1)
			{  
				cout << eval_VaR << ",";
			}else{
				cout << eval_VaR << "," << endl;
			}
		}
		cout << endl << "VaR query: " ;
		if (QUERY_TYPE)
		{
			for (int p =0; p < PATH_LENGTH; p++)
			{
				cout << "(" << query_state << "," <<map_policy[query_state] << ") ";		  
				best_pair = make_pair(query_state, map_policy[query_state]);
				if( fmdp->isOptimalAction(best_pair.first, best_pair.second))
					birl.addPositiveDemo(best_pair); 
				else  birl.addNegativeDemo(best_pair);
				query_state = fmdp->getNextState(query_state, map_policy[query_state]);
			}
		}
		else
		{
			cout << "(" << query_state << "," <<opt_policy[query_state] << ") ";    
			birl.addPositiveDemo(best_pair);
		}
		clock_t c_end = clock();
		cout << "\n[Timing] Time passed VaR: " << (c_end-c_start)*1.0 / CLOCKS_PER_SEC << " s\n";
		cout << endl << "------------"<< endl;

		// --- Active Entropy -----
		c_start = clock();
		map<unsigned int, double> entropies;
		cout << "Entropies: " << endl ;   
		for(unsigned int s = 0; s < numStates; s+= 1)
		{
			pair<unsigned int,unsigned int> state_action;
			unsigned int a;
			entropies.insert(make_pair(s,0.0));
			for( unsigned int a = 0 ; a < fmdp1->getNumActions(); a++) //
			{
				state_action = make_pair(s,a);
				double ent = gain_computer.getEntropy(state_action); 
				entropies[s] += ent;  
			}
			cout.precision(5);
			entropies[s] /= fmdp1->getNumActions();
			cout  << entropies[s] << ", " ;
			if (s % SIZE == SIZE - 1) cout << endl;
		}
		cout << endl;
		vector<pair<unsigned int, double> > argmax_entropies(entropies.begin(), entropies.end());
		sort(argmax_entropies.begin(), argmax_entropies.end(), less_second<unsigned int, double>());
		cout << "AS query: " ;          
		int curr_s = argmax_entropies[numStates-1].first;
		if (QUERY_TYPE)
		{
			for ( unsigned int idx = 0; idx < PATH_LENGTH ; ++idx)
			{
				cout << "(" << curr_s << "," <<map_policy1[curr_s] << "), ";		  
				best_pair = make_pair(curr_s ,map_policy1[curr_s]);	
				if(fmdp1->isOptimalAction(best_pair.first, best_pair.second))	
					birl1.addPositiveDemo(best_pair); 
				else
					birl1.addNegativeDemo(best_pair); 
				curr_s = fmdp->getNextState(curr_s, map_policy1[curr_s]);
			}
                        cout << endl;
		}
		else
		{
                        cout << argmax_entropies[numStates-1].first << endl;
			cout << argmax_entropies[numStates-1].first;
		 	best_pair = make_pair(argmax_entropies[numStates-1].first ,opt_policy[argmax_entropies[numStates-1].first]);
			birl1.addPositiveDemo(best_pair); 
		}
		c_end = clock();
		cout << "\n[Timing] Time passed Entropy: " << (c_end-c_start)*1.0 / CLOCKS_PER_SEC << " s\n";
		cout << endl << "------------"<< endl;

		// --- Active InfoGain -----
		c_start = clock();
		cout << "InfoGain:" << endl;
		cout << "MaxGain Path query: " ; 
		if(!QUERY_TYPE) // randomly sample if not doing critique query
		{
			int rand_ss = rand() % numStates;     
			cout << rand_ss << endl;
			best_pair = make_pair(rand_ss ,opt_policy[rand_ss]);	
			birl2.addPositiveDemo(best_pair);
		}
		else{ 
			vector< pair<unsigned int,unsigned int> > path;
			gain_computer2.GeneratePathQuery(SIZE, PATH_LENGTH, path);
			for(pair<unsigned int,unsigned int> state_action: path)
			{
				cout << "(" << state_action.first << "," << state_action.second << "), ";
				if(fmdp2->isOptimalAction(state_action.first, state_action.second))   
					birl2.addPositiveDemo(state_action);
				else
					birl2.addNegativeDemo(state_action);

			}
			cout << endl;

		}

		cout << endl;
		c_end = clock();
		cout << "\n[Timing] Time passed ARC: " << (c_end-c_start)*1.0 / CLOCKS_PER_SEC << " s\n";
		cout << endl << "------------"<< endl;

		//random
		c_start = clock();
		cout << "Random query: " ;          //current policy rollout
		unsigned int rand_s = rand() % numStates;
		if(QUERY_TYPE)
		{

			for ( unsigned int idx = 0; idx < PATH_LENGTH ; ++idx)
			{
				cout << "(" << rand_s << ", " << map_policy3[rand_s] << "), ";
				best_pair = make_pair(rand_s ,map_policy3[rand_s]);	
				if(fmdp3->isOptimalAction(best_pair.first, best_pair.second))	
					birl3.addPositiveDemo(best_pair); 
				else
					birl3.addNegativeDemo(best_pair); 
				rand_s = fmdp->getNextState(rand_s, map_policy3[rand_s]);

			}
		}
		else
		{
			cout << rand_s << endl;
			best_pair = make_pair(rand_s ,opt_policy[rand_s]);	
			birl3.addPositiveDemo(best_pair); 


		}
		cout << endl;
		c_end = clock();
		cout << "\n[Timing] Time passed random: " << (c_end-c_start)*1.0 / CLOCKS_PER_SEC << " s\n";
		cout << endl << "------------"<< endl;
	}

	birl.displayDemos();
	birl.run();

	birl1.displayDemos();
	birl1.run();

	birl2.displayDemos();
	birl2.run();

	birl3.displayDemos();
	birl3.run();

	cout.precision(4);

	FeatureGridMDP* mapMDP = birl.getMAPmdp();
	mapMDP->displayFeatureWeights();
	cout << "Recovered reward" << endl;
	mapMDP->displayRewards();

	vector<unsigned int> map_policy  (mapMDP->getNumStates());
	mapMDP->valueIteration(0.001);
	mapMDP->deterministicPolicyIteration(map_policy);
	mapMDP->displayPolicy(map_policy);
	double loss = policyLoss(map_policy, fmdp);

	FeatureGridMDP* mapMDP1 = birl1.getMAPmdp();
	mapMDP1->displayFeatureWeights();
	cout << "Recovered reward" << endl;
	mapMDP1->displayRewards();

	vector<unsigned int> map_policy1  (mapMDP1->getNumStates());
	mapMDP1->valueIteration(0.001);
	mapMDP1->deterministicPolicyIteration(map_policy1);
	mapMDP1->displayPolicy(map_policy1);
	double loss1 = policyLoss(map_policy1, fmdp1);

	FeatureGridMDP* mapMDP2 = birl2.getMAPmdp();
	mapMDP2->displayFeatureWeights();
	cout << "Recovered reward" << endl;
	mapMDP2->displayRewards();

	vector<unsigned int> map_policy2  (mapMDP2->getNumStates());
	mapMDP2->valueIteration(0.001);
	mapMDP2->deterministicPolicyIteration(map_policy2);
	mapMDP2->displayPolicy(map_policy2);
	double loss2 = policyLoss(map_policy2, fmdp2);

	FeatureGridMDP* mapMDP3 = birl3.getMAPmdp();
	mapMDP3->displayFeatureWeights();
	cout << "Recovered reward" << endl;
	mapMDP3->displayRewards();

	vector<unsigned int> map_policy3  (mapMDP3->getNumStates());
	mapMDP3->valueIteration(0.001);
	mapMDP3->deterministicPolicyIteration(map_policy3);
	mapMDP3->displayPolicy(map_policy3);
	double loss3 = policyLoss(map_policy3, fmdp3);

	cout.precision(4);
	cout << "Current ActiveVaR policy loss: "  << loss << "%" << endl;
	losses[INTERACTIONS] = loss;
	cout << "Current AS policy loss1: "  << loss1 << "%" << endl;
	losses1[INTERACTIONS] = loss1;
	cout << "Current ARC policy loss2: "  << loss2 << "%" << endl;
	losses2[INTERACTIONS] = loss2;
	cout << "Current Random policy loss3: "  << loss3 << "%" << endl;
	losses3[INTERACTIONS] = loss3;

	cout << "ActiveVaR Losses:";
	for(unsigned int i =0 ; i < INTERACTIONS + 1; i++) cout << losses[i] << "," ;
	cout << endl;

	cout << "AS Losses1:";
	for(unsigned int i =0 ; i < INTERACTIONS + 1; i++) cout << losses1[i] << "," ;
	cout << endl;

	cout << "ARC Losses2:";
	for(unsigned int i =0 ; i < INTERACTIONS + 1; i++) cout << losses2[i] << "," ;
	cout << endl;

	cout << "Random Losses3:";
	for(unsigned int i =0 ; i < INTERACTIONS + 1; i++) cout << losses3[i] << "," ;
	cout << endl;

	//clean up
	double** stateFeatures = fmdp->getStateFeatures();
	//delete features
	for(unsigned int s1 = 0; s1 < fmdp->getNumStates(); s1++)
	{
		delete[] stateFeatures[s1];
	}
	delete[] stateFeatures;

	delete fmdp;


}


