
#ifndef gain_h
#define gain_h

#include <cmath>
#include <stdlib.h>
#include <vector>
#include <map>

#include <iostream>
#include <fstream>
#include <limits>

#include "../include/mdp.hpp"
#include "../include/feature_birl.hpp"


// Fill the zipped vector with pairs consisting of the
// corresponding elements of a and b. (This assumes 
// that the vectors have equal length)
using namespace std;

void zip(
		long double* a, 
		long double* b,
		unsigned int size,
		vector<pair<long double,long double> > &zipped)
{
	zipped.resize(size);
	for(unsigned int i=0; i<size; ++i)
	{
		zipped[i] = make_pair(a[i], b[i]);
	}
}

// Write the first and second element of the pairs in 
// the given zipped vector into a and b. (This assumes 
// that the vectors have equal length)

void unzip(
		const vector<pair<long double, long double> > &zipped, 
		long double* a, 
		long double* b)
{
	for(unsigned int i=0; i< zipped.size(); i++)
	{
		a[i] = zipped[i].first;
		b[i] = zipped[i].second;
	}
}

bool greaterThan(const pair<long double, long double>& a, const pair<long double, long double>& b)
{
	return a.first < b.first;
}

class FeatureInfoGainCalculator{

	private:
		FeatureBIRL * base_birl = nullptr;
		FeatureBIRL * good_birl = nullptr;
		FeatureBIRL * bad_birl = nullptr;
		FeatureGridMDP* curr_mdp;
		double min_r, max_r, step_size, alpha;
		unsigned int chain_length;


	public:
		FeatureInfoGainCalculator(FeatureBIRL* input_birl) {
			min_r = input_birl->getMinReward();
			max_r = input_birl->getMaxReward();
			curr_mdp = input_birl->getMDP();
			step_size = input_birl->getStepSize();
			chain_length = input_birl->getChainLength();
			alpha = input_birl->getAlpha();
			base_birl = input_birl; 
			good_birl = new FeatureBIRL(curr_mdp, min_r, max_r, chain_length, step_size, alpha);
			bad_birl  = new FeatureBIRL(curr_mdp, min_r, max_r, chain_length, step_size, alpha);
		};

		~ FeatureInfoGainCalculator(){
			if(good_birl != nullptr) delete good_birl;
			if(bad_birl != nullptr) delete bad_birl;
		};

		long double getInfoGain(pair<unsigned int ,unsigned int> state_action);
		long double getInfoGainFromSamples(pair<unsigned int,unsigned int> state_action, double * prob_good);

		long double KLdivergence(long double* p, long double* q, unsigned int size);
		long double KNN_KLdivergence(FeatureGridMDP** p, FeatureGridMDP** q, unsigned int size);

		long double Entropy(double* p, unsigned int size);
		long double JSdivergence(long double* p, long double* q, unsigned int size);
		void sortAndWriteToFile(long double * base_posterior,long double * good_posterior, long double* bad_posterior);

		long double getEntropy(pair<unsigned int,unsigned int> state_action, int K = 10);

		void GeneratePathQuery(unsigned int num_paths, unsigned int path_length, vector<pair<unsigned int, unsigned int>> & best_path);


};

void FeatureInfoGainCalculator::GeneratePathQuery(unsigned int num_paths, unsigned int path_length, vector<pair<unsigned int, unsigned int>> & best_path)
{
	FeatureGridMDP** R_chain_base = base_birl->getRewardChain();
	long double max_gain = 0;
	vector<vector<pair<unsigned int, unsigned int>>> paths;

	// Generate Paths
	for (unsigned int p=0; p < num_paths; p++)
	{
		vector<pair<unsigned int, unsigned int>> curr_path;
		unsigned int sample_idx = rand()%chain_length;

		cout << "    Selected R" << sample_idx << endl;
		FeatureGridMDP* sample_mdp = R_chain_base[sample_idx];
		vector<unsigned int> map_policy (sample_mdp->getNumStates());
		sample_mdp->deterministicPolicyIteration(map_policy);

		unsigned int curr_s = rand()%sample_mdp->getNumStates();
		unsigned int curr_a = map_policy[curr_s];
		unsigned int prev_s = curr_s;
		unsigned int next_s = sample_mdp->getNextState(curr_s, curr_a);  

		cout << "     - Path" << p << ": ";
		unsigned int i;
		for(i = 0; i < path_length; i++)
		{
			cout << "(" << curr_s << "," << curr_a << ") ";
			curr_path.push_back(make_pair(curr_s,curr_a));
			if(next_s == curr_s || next_s == prev_s || sample_mdp->isTerminalState(curr_s)) break; 
			prev_s = curr_s;
			curr_s = next_s;
			curr_a = map_policy[curr_s];
			next_s = sample_mdp->getNextState(curr_s, curr_a);  
		}
		cout << endl;
		if (i > path_length / 2) paths.push_back(curr_path);
	}

	//Calculate info gains 
	map<unsigned int, long double> info_gains;
	map<unsigned int, double> probs_good;

	for (unsigned int p=0; p < paths.size(); p++)
	{
		vector<pair<unsigned int, unsigned int>> curr_path = paths[p];
		if(!base_birl->isDemonstration(curr_path[0]))
		{
			double prob_good;
			long double info_gain = getInfoGainFromSamples(curr_path[0], &prob_good);
			info_gains.insert(make_pair(p,info_gain));
			probs_good.insert(make_pair(p,prob_good));
		}else{
			info_gains.insert(make_pair(p,0.0l));
			probs_good.insert(make_pair(p,0.0));
		}
		cout << " Initial Info Gain: " << info_gains[p] << endl;
	}


	for (unsigned int p=0; p < paths.size(); p++)
	{
		vector<pair<unsigned int, unsigned int>> curr_path = paths[p];
		vector<pair<unsigned int, unsigned int>> assumptions;
		for( unsigned int i = 1; i < curr_path.size(); i++)
		{
			if(!base_birl->isDemonstration(curr_path[i-1]))
			{
				if(probs_good[p] > 0.5)
				{
					base_birl->addPositiveDemo(curr_path[i-1]); 
				}else{
					base_birl->addNegativeDemo(curr_path[i-1]);
				}
				assumptions.push_back(curr_path[i-1]);
				base_birl->run();
			}

			if(!base_birl->isDemonstration(curr_path[i]))
			{
				double prob_good;
				info_gains[p] += getInfoGainFromSamples(curr_path[i], &prob_good)*pow(0.9,i);
				probs_good[p] = prob_good;
			}
		}

		for( unsigned int i = 0; i < assumptions.size(); i++) base_birl->removeDemonstration(assumptions[i]);

		if (info_gains[p] > max_gain)
		{
			max_gain = info_gains[p];
			best_path.clear();
			for(unsigned int i = 0; i < curr_path.size(); i++) best_path.push_back(curr_path[i]);
		}
	}
	cout << ">>>> Max Info Gain: " << max_gain << endl;

}

long double FeatureInfoGainCalculator::getEntropy(pair<unsigned int,unsigned int> state_action, int K)
{
	long double info_gain = 0.0;

	unsigned int state = state_action.first;
	unsigned int action = state_action.second;

	long double probability_good = 0; 

	vector<double> frequencies;
	double total_frequency = 0;
	for(unsigned int i = 0; i < K; i++) frequencies.push_back(0);

	FeatureGridMDP** R_chain_base = base_birl->getRewardChain();
	for(unsigned int i= 50; i < chain_length; i++)
	{
		FeatureGridMDP* temp_mdp = R_chain_base[i];
		unsigned int numActions = temp_mdp->getNumActions();
		double Z0 [numActions]; 
		for(unsigned int a = 0; a < numActions; a++) Z0[a] = alpha*temp_mdp->getQValue(state,a);
		probability_good = exp(alpha*temp_mdp->getQValue(state,action) - base_birl->logsumexp(Z0,numActions));
		for (unsigned int i = 0; i < K; i++)
		{
			if(probability_good > (double)i/K && probability_good <= (double)(i+1)/K )
			{
				frequencies[i] += 1;
				total_frequency += 1;
				break;
			}
		}
	}

	for(unsigned int i=0; i < K; i++)
	{
		if(frequencies[i] != 0){
			frequencies[i] /= total_frequency;
			info_gain += -(frequencies[i]*log(frequencies[i]));
		}
	}

	return info_gain;

}


long double FeatureInfoGainCalculator::getInfoGainFromSamples(pair<unsigned int,unsigned int> state_action, double * prob_good)
{
	long double info_gain = 0.0;

	unsigned int state = state_action.first;
	unsigned int action = state_action.second;

	good_birl->addPositiveDemos(base_birl->getPositiveDemos());
	good_birl->addNegativeDemos(base_birl->getNegativeDemos());
	good_birl->addPositiveDemo(state_action);
	good_birl->setAlpha(alpha);

	good_birl->run();

	bad_birl->addPositiveDemos(base_birl->getPositiveDemos());
	bad_birl->addNegativeDemos(base_birl->getNegativeDemos());
	bad_birl->addNegativeDemo(state_action);
	bad_birl->setAlpha(alpha);

	bad_birl->run();

	FeatureGridMDP** R_chain_base = base_birl->getRewardChain();
	FeatureGridMDP** R_chain_good = good_birl->getRewardChain();
	FeatureGridMDP** R_chain_bad  = bad_birl->getRewardChain();

	long double probability_good = 0;
	for(unsigned int i=0; i < chain_length; i++)
	{
		MDP* temp_mdp = R_chain_base[i];
		unsigned int numActions = temp_mdp->getNumActions();

		double Z0 [numActions]; 
		for(unsigned int a = 0; a < numActions; a++) Z0[a] = alpha*temp_mdp->getQValue(state,a);
		probability_good += exp(alpha*temp_mdp->getQValue(state,action) - base_birl->logsumexp(Z0,numActions));
	}
	probability_good /= chain_length;
	*prob_good = probability_good;
	long double divergence_good = KNN_KLdivergence(R_chain_base, R_chain_good, chain_length);
	long double divergence_bad  = KNN_KLdivergence(R_chain_base, R_chain_bad, chain_length);

	info_gain = divergence_good*probability_good + (1 - probability_good)*divergence_bad;

	good_birl->removeAllDemostrations();
	bad_birl->removeAllDemostrations();
	return info_gain;

}


long double FeatureInfoGainCalculator::getInfoGain(pair<unsigned int,unsigned int> state_action)
{
	long double info_gain = 0.0;

	unsigned int state = state_action.first;
	unsigned int action = state_action.second;

	good_birl->addPositiveDemos(base_birl->getPositiveDemos());
	good_birl->addNegativeDemos(base_birl->getNegativeDemos());
	good_birl->addPositiveDemo(state_action);
	good_birl->setAlpha(alpha);

	good_birl->run();

	bad_birl->addPositiveDemos(base_birl->getPositiveDemos());
	bad_birl->addNegativeDemos(base_birl->getNegativeDemos());
	bad_birl->addNegativeDemo(state_action);
	bad_birl->setAlpha(alpha);

	bad_birl->run();

	FeatureGridMDP** R_chain_base = base_birl->getRewardChain();
	FeatureGridMDP** R_chain_good = good_birl->getRewardChain();
	FeatureGridMDP** R_chain_bad  = bad_birl->getRewardChain();

	long double probability_good = 0;
	for(unsigned int i=0; i < chain_length; i++)
	{
		MDP* temp_mdp = R_chain_base[i];
		unsigned int numActions = temp_mdp->getNumActions();

		double Z0 [numActions]; 
		for(unsigned int a = 0; a < numActions; a++) Z0[a] = alpha*temp_mdp->getQValue(state,a);
		probability_good += exp(alpha*temp_mdp->getQValue(state,action) - base_birl->logsumexp(Z0,numActions));
	}
	probability_good /= chain_length;
	long double divergence_good = KNN_KLdivergence(R_chain_base, R_chain_good, chain_length);
	long double divergence_bad  = KNN_KLdivergence(R_chain_base, R_chain_bad, chain_length);

	info_gain = divergence_good*probability_good + (1 - probability_good)*divergence_bad;

	good_birl->removeAllDemostrations();
	bad_birl->removeAllDemostrations();
	return info_gain;
}

long double FeatureInfoGainCalculator::KNN_KLdivergence(FeatureGridMDP** p, FeatureGridMDP** q, unsigned int size)
{
	long double divergence = 0.0;
	long double c = (long double)(p[0]->getNumStates()) / size;

	for(unsigned int i=0; i < size; i++)
	{
		long double dist_pp = numeric_limits<double>::infinity();
		long double dist_pq = numeric_limits<double>::infinity();

		for(unsigned int j=0; j < size; j++)
		{
			if(j != i)  
			{
				long double dist_ij = p[i]->L2_distance(p[j]);
				if (dist_ij < dist_pp )  dist_pp = dist_ij;
			}
		}

		for(unsigned int j=0; j < size; j++)
		{
			long double dist_ij = p[i]->L2_distance(q[j]);
			if (dist_ij < dist_pq)  dist_pq = dist_ij;
		}

		dist_pp = max(dist_pp, 0.000001l);
		dist_pq = min(dist_pq, 10000.0l);
		divergence += log((dist_pq/dist_pp)/1000+1);
		if (divergence < 0){ 
			divergence = 100000.0l;
			break;
		}  
	}


	divergence = c*(divergence+log(1000)) + log(size/(size-1));
	return divergence;
}

long double FeatureInfoGainCalculator::KLdivergence(long double* p, long double* q, unsigned int size)
{
	long double divergence = 0.0;
	for(unsigned int i=0; i < size; i++)
	{
		if(q[i] > 0.00000001 && p[i] != 0) divergence += (p[i]*log(p[i]/q[i]));
		else if(p[i] != 0) divergence += (p[i]*log(p[i]*100000000));
	}

	return divergence;
}

long double FeatureInfoGainCalculator::Entropy(double* p, unsigned int size)
{
	long double entropy = 0.0;
	for(unsigned int i=0; i < size; i++)
	{
		if(p[i] != 0) entropy -= (p[i]*log(p[i]));
	}
	return entropy;
}

long double FeatureInfoGainCalculator::JSdivergence(long double* p, long double* q, unsigned int size)
{
	long double divergence = 0.0;   

	for(unsigned int i=0; i < size; i++)
	{
		long double avg = (p[i] + q[i]) /2;
		if     (avg != 0 && p[i] == 0 && q[i] != 0) divergence += 0.5*(q[i]*log(q[i]/avg));
		else if(avg != 0 && p[i] != 0 && q[i] == 0) divergence += 0.5*(p[i]*log(p[i]/avg));
		else if(avg != 0 && p[i] != 0 && q[i] != 0) divergence += 0.5*(p[i]*log(p[i]/avg)) + 0.5*(q[i]*log(q[i]/avg));
	}

	return divergence;
}

void FeatureInfoGainCalculator::sortAndWriteToFile(long double * base_posterior,long double * good_posterior, long double* bad_posterior)
{
	//for plotting purpose
	vector<pair<long double, long double> > zipped_bg, zipped_bb;
	zip(base_posterior, good_posterior, chain_length, zipped_bg);
	zip(base_posterior, bad_posterior,  chain_length, zipped_bb);

	sort(zipped_bg.begin(), zipped_bg.end(), greaterThan);
	sort(zipped_bb.begin(), zipped_bb.end(), greaterThan);

	unzip(zipped_bg, base_posterior, good_posterior);
	unzip(zipped_bb, base_posterior, bad_posterior);

	//printing for plot
	//cout << "\n--- printing for plot ---" << endl;
	//sort(base_posterior.begin(), base_posterior.end());
	ofstream basefile;
	basefile.open ("data/base.dat");
	for(unsigned int idx=0; idx < chain_length; idx++){
		basefile << idx << " " << base_posterior[idx] << endl;
	}
	basefile.close();

	//sort(good_posterior.begin(), good_posterior.end());
	ofstream goodfile;
	goodfile.open ("data/good.dat");
	for(unsigned int idx=0; idx < chain_length; idx++){
		goodfile << idx << " " << good_posterior[idx] << endl;
	}
	goodfile.close();

	//sort(bad_posterior.begin(), bad_posterior.end());
	ofstream badfile;
	badfile.open ("data/bad.dat");
	for(unsigned int idx=0; idx < chain_length; idx++){
		badfile << idx << " " << bad_posterior[idx] << endl;
	}
	badfile.close();
}


#endif
