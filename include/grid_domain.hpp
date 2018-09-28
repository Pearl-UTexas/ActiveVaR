#ifndef grid_domains_h
#define grid_domains_h

// set up random gridworld
double** initRandomFeaturesRandomDomain(const int numStates, const int numFeatures)
{
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
    {
        stateFeatures[i] = new double[numFeatures];
        double feature[numFeatures];
        for (int feat=0; feat < numFeatures; feat++) feature[feat] = rand()%100/100.0; 
        std::copy(feature, feature+numFeatures, stateFeatures[i]);
    }
    return stateFeatures;
}

//Set up stateFeatures for simple grid world 
double** initFeaturesToyFeatureDomain5x5(const int numStates, const int numFeatures)
{
     if(numStates != 25 || numFeatures != 5) 
     {
        cout << "[ERROR] This domain only works for 5x5 with 5 features!" << endl;
        return nullptr;
     }
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double whiteFeature[]  = {1,0,0,0,0};
    double redFeature[]    = {0,1,0,0,0};
    double blueFeature[]   = {0,0,1,0,0};
    double yellowFeature[] = {0,0,0,1,0};
    double greenFeature[]  = {0,0,0,0,1};
    
    double YBFeature[]    = {0,0,1,1,0};

    char features[] = {'w','w','w','b','w',
                       'w','w','g','b','w',
                       'w','w','w','b','w',
                       'y','y','w','b','w',
                       'w','w','w','b','w',
                       };
                       
    for(int i=0; i < numStates; i++)
    {
        switch(features[i])
        {
            case 'w':
                std::copy(whiteFeature, whiteFeature+numFeatures, stateFeatures[i]);
                break;
            case 'r':
                std::copy(redFeature, redFeature+numFeatures, stateFeatures[i]);
                break;
            case 'b':
                std::copy(blueFeature, blueFeature+numFeatures, stateFeatures[i]);
                break;
            case 'y':
                std::copy(yellowFeature, yellowFeature+numFeatures, stateFeatures[i]);
                break;
            case 'g':
                std::copy(greenFeature, greenFeature+numFeatures, stateFeatures[i]);
                break;
            case 'Y':
                std::copy(YBFeature, YBFeature+numFeatures, stateFeatures[i]);
                break;
        }
    
    }
    return stateFeatures;
}




#endif

