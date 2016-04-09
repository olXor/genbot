#ifndef GENOME_HEADER_FILE
#define GENOME_HEADER_FILE

#include "cluster.h"
#include <time.h>
#include <stdlib.h>
#include <random>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
//#include "nvwa/debug_new.h"

struct ConvolutionProperties {
    int rank;
    std::vector<int> dimensions;
    int inputRangeBegin;
    int inputRangeEnd;
    std::vector<int> inputSpaceDimensions;
    int numLayers;
    int nodesPerLayer;
    double stepfactor;
    double transferWidth;
};

static ConvolutionProperties CONVOLUTION_PROPERTIES_INITIALIZER = {
    1, {1}, 0, 0, {1}, 1, 1, 1, 1
};

class Genome {
    private:
        size_t CHILDDEPTH = 1;

        int ALLOW_SIDE_WEIGHTS = 0;
        int ALLOW_SIDE_MEMS = 0;

        size_t MAX_NODESPERLAYER = 200;
        size_t MAX_LAYERS = 2;
        size_t MAX_NUMPERTURBS = 0;

        //"convolution levels" refers to the number of sequential convolutions the data passes through before going to the cluster. "convolution node layers" refers to the number of node layers in each convolution.
        size_t MAX_CONVOLUTION_LEVELS = 2;
        size_t MIN_CONVOLUTION_LEVELS = 2;

        size_t MAX_CONVOLUTIONS = 3;
        size_t MIN_CONVOLUTIONS = 1;

        size_t MAX_CONVOLUTION_NODE_LAYERS = 1;
        size_t MAX_CONVOLUTION_NODESPERLAYER = 10;

        size_t MAX_CONVOLUTION_DIMENSION = 10;
        size_t MIN_CONVOLUTION_DIMENSION = 2;

        double CONVOLUTION_DIMENSION_LAYER_MULTIPLIER = 2.0;

        size_t NUM_TURNS_SAVED = 15;

    public:
        ClusterParameters** pars;   //length CHILDDEPTH
        std::vector<std::vector<ConvolutionProperties>> convProperties;
        ConvolutionProperties defaultConvProp;

        Genome* copy();
        Genome* mate(Genome* partner);

        int getChildDepth() {
            return CHILDDEPTH;
        }

        void createRandomGenome();

        Genome() {
            pars = new ClusterParameters*[CHILDDEPTH];
            for(size_t i=0; i<CHILDDEPTH; i++) {
                pars[i] = NULL;
            }
            extraAnswerTurns = 0;
            numPerturbRuns = 0;
            perturbChance = 0;
            perturbFactor = 0;
            defaultConvProp = CONVOLUTION_PROPERTIES_INITIALIZER;
            readCfg();
        }

        Genome(ConvolutionProperties def) {
            pars = new ClusterParameters*[CHILDDEPTH];
            for(size_t i=0; i<CHILDDEPTH; i++) {
                pars[i] = NULL;
            }
            extraAnswerTurns = 0;
            numPerturbRuns = 0;
            perturbChance = 0;
            perturbFactor = 0;
            defaultConvProp = def;
            readCfg();
        }

        ~Genome() {
            for(size_t i=0; i<CHILDDEPTH; i++) {
                if(pars[i] != NULL)
                    delete pars[i];
            }
            delete [] pars;
        }

        ConvolutionProperties getDefaultConvProp() {
            return defaultConvProp;
        }

        void setDefaultConvProp(ConvolutionProperties cp) {
            defaultConvProp = cp;
        }

        int getExtraAnswerTurns() {
            return extraAnswerTurns;
        }

        void setExtraAnswerTurns(int eat) {
            extraAnswerTurns = eat;
        }

        int getNumPerturbRuns() {
            return numPerturbRuns;
        }

        void setNumPerturbRuns(int npr) {
            numPerturbRuns = npr;
        }

        double getPerturbChance() {
            return perturbChance;
        }

        void setPerturbChance(double pc) {
            perturbChance = pc;
        }

        double getPerturbFactor() {
            return perturbFactor;
        }

        void setPerturbFactor(double pf) {
            perturbFactor = pf;
        }

        std::vector<int> getNumConvolutionTypes() {
            return numConvolutionTypes;
        }

        void setNumConvolutionTypes(std::vector<int> n) {
            numConvolutionTypes = n;
        }

        void setNumConvolutionTypes(int layer, int n) {
            numConvolutionTypes[layer] = n;
        }

        void resizeNumConvolutionTypes(int size) {
            numConvolutionTypes.resize(size);
        }

        void addNumConvolutionTypesLayer(int n) {
            numConvolutionTypes.push_back(n);
        }

        void eraseNumConvolutionTypesLayer() {
            numConvolutionTypes.pop_back();
        }

        int getNumConvolutionLayers() {
            return numConvolutionLayers;
        }

        void setNumConvolutionLayers(int n) {
            numConvolutionLayers = n;
        }

        int getNumTurnsSaved() {
            return NUM_TURNS_SAVED;
        }

        bool hasSideWeights();

        void saveGenome(const char* file);
        void loadGenome(const char* file);

    private:
        bool mutate();
        bool raremutate();
        bool coinflip();
        double gaussianRandom(double in, double stdev);
        ClusterParameters* createRandomClusterParameters();
        int getRandomNumLayers();
        int getRandomNodesPerLayer();
        double getRandomStepFactor();
        double getRandomMemFactor();
        double getRandomMemNorm();
        int getRandomLearnStyleSide();
        int getRandomExtraAnswerTurns();
        int getRandomNumPerturbRuns();
        double getRandomPerturbChance();
        double getRandomPerturbFactor();
        double getRandomTransferWidth();
        std::vector<int> getRandomConvolutionDimensions(int rank, int level);
        int getRandomConvolutionNumLayers();
        int getRandomConvolutionNodesPerLayer();
        double getRandomConvolutionStepFactor();
        double getRandomConvolutionTransferWidth();
        int getRandomNumConvolutionLayers();
        std::vector<int> getRandomNumConvolutionTypes(int numLevels);
        ConvolutionProperties mutateConvolutionProperties(ConvolutionProperties cp);
        ConvolutionProperties getRandomConvolutionProperties(int level);
        std::vector<ConvolutionProperties> getRandomConvolutionPropertiesLayer(int level);
        int getRandomNumConvolutions();

        void readCfg();

        int extraAnswerTurns;
        int numPerturbRuns;
        double perturbChance;
        double perturbFactor;
        int numConvolutionLayers;
        std::vector<int> numConvolutionTypes;
};
#endif
